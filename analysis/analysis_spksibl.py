import os
from labdatatools.analysis import *
import argparse

from spks.spikeglx_utils import read_spikeglx_meta

def recursive_map(f, it):
    return (recursive_map(f, x) if isinstance(x, tuple) else f(x) for x in it)
        
class AnalysisSpksibl(BaseAnalysisPlugin):
    def __init__(self,subject,
                 session = None,
                 datatypes = [''],
                 includes = [''],
                 excludes = default_excludes,
                 bwlimit = None,
                 overwrite = False,
                 delete_session = False,
                 **kwargs):
        '''
labdatatools wrapper for running spike sorting through spks
Joao Couto - October 2023
        '''
        super(AnalysisSpksibl,self).__init__(
            subject = subject,
            session = session,
            datatypes = datatypes,
            includes = includes,
            excludes = excludes,
            bwlimit = bwlimit,
            overwrite = overwrite,
            delete_session = delete_session,
            **kwargs)
        self.name = 'spksibl'
        self.output_folder = 'spike_sorters/kilosort2.5'
        self.datatypes = ['raw_ephys_data']
        if not datatypes == ['']:
            self.input_folder = datatypes[0]
        else:
            self.input_folder = 'raw_ephys_data'

    def parse_arguments(self,arguments = []):
        parser = argparse.ArgumentParser(
            description = 'Analysis of spike data using kilosort version 2.5 through spks.',
            usage = 'spks -- <PARAMETERS>')
        
        parser.add_argument('-p','--probe',
                            action='store', default=None, type = int,
                            help = "THIS DOES NOTHING NOW. WILL BE FOR OPENING PHY")
        parser.add_argument('-t','--tempdir',
                            action='store', default = '/scratch', type = str,
                            help = "Temporary directory to store intermediate results. (default is /scratch - needs to be a fast disk like an NVME)")
        parser.add_argument('-d','--device',
                            action='store', default = 'cuda', type = str,
                            help = "Device for pytorch (cuda|cpu)")

        args = parser.parse_args(arguments[1:])
        self.probe = args.probe
        self.tempdir = args.tempdir
        self.device = args.device

    def _run(self):
        from spks.sorting import ks25_sort_multiprobe_sessions
        from spks.io import list_spikeglx_binary_paths
        from spks.spikeglx_utils import read_geommap, read_spikeglx_meta
        import subprocess
        from pathlib import Path
        import re

        folders = self.get_sessions_folders()

        # The IBL pipeline splits 4 shank recordings into separate binary files before sorting
        # We need to acount for this here and update the meta files to sort them separately. We will treat each shank as a probe
         
        # ##### 1. rename the files for individual shanks
        binfiles = []
        for f in folders:
            binfiles.extend(list(Path(f).rglob('*[0-9].ap.meta')))
        for file in binfiles:
            prbfolder = file.parent 
            finalchar = str(prbfolder)[-1]
            prbnumber = str(prbfolder)[-2]
            if finalchar in ['a','b','c','d']: #if shanks were split by IBL
                files = prbfolder.glob('*')
                for old_filename in files:
                    new_name = re.sub(f'imec[0-9][.]',f'imec{prbnumber}{finalchar}.', old_filename.name) 
                    new_filename = old_filename.parent / new_name
                    old_filename.replace(new_filename) # rename the files
                    old_filename.unlink(missing_ok=True)

        ####### 2. replace the missing tilde in the files for proper parsing
        tmp = [list_spikeglx_binary_paths(s) for s in folders]
        print(tmp)
        assert len(tmp) == 1 #hacky, only works with one date right now
        replace_patterns = ['imroTbl','muxTbl','snsChanMap','snsGeomMap'] 
        for file in tmp[0]: #add back the tilde that was removed by IBL pipeline
            file = file[0].replace('.ap.cbin','.ap.meta')
            with open(file) as f:
                contents = str(f.read())
            for patt in replace_patterns:
                if f'~{patt}' not in contents:
                    command = f'sed -i s/{patt}/~{patt}/g {file}'
                    res = subprocess.check_output(command, shell=True).decode()

        ####### 3. fix the imro table and snsgeommap from the meta files
        LETTER_TO_SHANK = dict(a=0, b=1, c=2, d=3)
        
        for file in tmp[0]:
            if str(Path(file[0]).parent)[-1] not in ['a','b','c','d']: # skip over probes that havent been split by IBL
                continue
            file = file[0].replace('.ap.cbin','.ap.meta')
            meta = read_spikeglx_meta(file)
            geommap, _= read_geommap(meta['snsGeomMap'])

            regex = re.compile('imec[0-9][a-z]')
            probe_name = re.search(regex,file)[0]
            shank_letter = probe_name[-1]
            shank_num = LETTER_TO_SHANK[shank_letter]

            recorded_channels = (geommap['shank_id'] == shank_num).values
            recorded_channels = np.where(recorded_channels)[0]

            geom = [meta['snsGeomMap'][0]] + [meta['snsGeomMap'][i+1] for i in recorded_channels]
            imro = [meta['imroTbl'][0]] + [meta['imroTbl'][i+1] for i in recorded_channels]
            #mux = [meta['muxTbl'][0]] + [meta['muxTbl'][i+1] for i in recorded_channels]
            chan = [meta['snsChanMap'][0]] + [meta['snsChanMap'][i+1] for i in recorded_channels]

            new_geom_string = '~snsGeomMap=' + ''.join([f'({i})' for i in geom]) + '\n'
            new_imro_string = '~imroTbl=' + ''.join([f'({i})' for i in imro]) + '\n'
            #new_mux_string = '~muxTbl=' + ''.join([f'({i})' for i in mux]) + '\n'
            new_chanmap_string = '~snsChanMap=' + ''.join([f'({i})' for i in chan]) + '\n'
            
            with open(file,'r') as f:
                text = f.readlines()
                imro_line_ind = np.where(['imroTbl' in line for line in text])[0][0]
                snsmap_line_ind = np.where(['snsGeomMap' in line for line in text])[0][0]
                mux_line_ind = np.where(['muxTbl' in line for line in text])[0][0]
                chanmap_line_ind = np.where(['snsChanMap' in line for line in text])[0][0]

                text[imro_line_ind] = new_imro_string
                text[snsmap_line_ind] = new_geom_string
                #text[mux_line_ind] = new_mux_string
                text[chanmap_line_ind] = new_chanmap_string
            with open(file, 'w') as ff:
                ff.writelines(text)
        

        results = ks25_sort_multiprobe_sessions(folders,
                                                temporary_folder=self.tempdir,
                                                use_docker=False,
                                                device = self.device,
                                                sorting_results_path_rules=['..', '..', '{sortname}', '{probename}'],
                                                sorting_folder_dictionary={'sortname': self.output_folder,
                                                                           'probename': 'probe0'})
   
        if not len(results):
            self.upload = False

    def validate_parameters(self):
        if len(self.subject)>1:
            raise(ValueError('Specify only one subject.'))
        if not len(self.subject[0]) or not len(self.session):
            raise(ValueError('Specify a subject and a session (-a <SUBJECT> -s <SESSION>).'))

