import os
from labdatatools.analysis import *
import argparse

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
        folders = self.get_sessions_folders()
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

