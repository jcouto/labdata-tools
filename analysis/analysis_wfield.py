from labdatatools.analysis import *
import argparse

class AnalysisWfield(BaseAnalysisPlugin):
    def __init__(self,subject,
                 session = None,
                 datatypes = [''],
                 includes = [''],
                 excludes = default_excludes,
                 bwlimit = None,
                 overwrite = False,
                 **kwargs):
        '''
labdatatools wrapper for running wfield pre-processing.

Joao Couto - 2021
        '''
        super(AnalysisWfield,self).__init__(
            subject = subject,
            session = session,
            datatypes = datatypes,
            includes = includes,
            excludes = excludes,
            bwlimit = bwlimit,
            overwrite = overwrite,
            **kwargs)
        
        self.name = 'wfield'
        self.datatypes = ['one_photon']
        if not datatypes == ['']:
            self.input_folder = datatypes[0]
        else:
            self.input_folder = 'one_photon'
        self.output_folder = 'wfield'
        
    def parse_arguments(self,arguments = []):
        parser = argparse.ArgumentParser(
            description = 'Analysis widefield data.',
            usage = 'wfield -- <PARAMETERS>')
        
        parser.add_argument('--raw',
                            action='store_true', default=False,
                            help = "Open the GUI to explore data or check landmarks")
        parser.add_argument('--open',
                            action='store_true', default=False,
                            help = "Open the GUI to explore data or check landmarks")
        
        args = parser.parse_args(arguments[1:])
        self.gui_raw = args.raw
        self.gui_open = args.open
        if args.open or args.raw:
            self.has_gui = True
        
    def _run(self):
        folders = self.get_sessions_folders()
        print(folders)
        for folder in folders:
            f = glob(pjoin(folder,self.input_folder))
            if len(f):
                f = f[0]
                outfolder = pjoin(folder,self.output_folder)
                if self.gui_raw:
                    cmd = 'wfield open_raw {0}'.format(f)
                elif self.gui_open:
                    cmd = 'wfield open {0}'.format(outfolder)
                else:
                    cmd = 'wfield preprocess {0} -o {1}'.format(f,outfolder)
                os.system(cmd)
                if not os.path.exists(outfolder):
                    self.upload = False
        
    def validate_parameters(self):
        if len(self.subject)>1:
            raise(ValueError('Specify only one subject.'))
        if not len(self.subject[0]) or not len(self.session):
            raise(ValueError('Specify a subject and a session (-a <SUBJECT> -s <SESSION>).'))

