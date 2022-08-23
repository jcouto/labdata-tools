from labdatatools import BaseAnalysisPlugin
import argparse
from glob import glob
from os.path import join as pjoin
import os

class AnalysisWfield(BaseAnalysisPlugin):
    def __init__(self,subject,
                 session = None,
                 datatypes = [''],
                 includes = [''],
                 excludes = [''],
                 bwlimit = None,
                 overwrite = False,
                 **kwargs):
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
            self.input_folder = datatypes
        self.output_folder = 'wfield'
        self.camera = camera
        
    def parse_arguments(self,arguments = []):
        parser = argparse.ArgumentParser(
            description = 'Analysis widefield data.',
            usage = 'wfield -- <PARAMETERS>')
        
        parser.add_argument('--open',
                            action='store_true', default=False, type=bool,
                            help = "Open the GUI to explore data or check landmarks")
        

        args = parser.parse_args(arguments[1:])
        
        self.gui = args.open
        
    def _run(self):
        folders = self.get_sessions_folders()
        # find the camera folder
        for folder in folders:
            for datatype in self.datatypes:
                f = glob(pjoin(folder,datatype,'*'+self.camera+'*.avi'))
                if len(f):
                    f = f[0]
                    fname = os.path.basename(f)
                    fname = fname.replace('.avi',self.output_extension)
                    fname = pjoin(folder,self.output_folder,fname)
                    param = ''
                    print(fname)
                    if os.path.exists(fname):
                        param += ' -p {0}'.format(fname.replace('.h5','.json'))
                    cmd = 'mptracker-gui {0} -o {1}{2}'.format(f,fname,param)
                    print(cmd)
                    os.system(cmd)
                    if not os.path.exists(fname):
                        self.upload = False
        
        
    def validate_parameters(self):
        if len(self.subject)>1:
            raise(ValueError('Specify only one subject.'))
        if not len(self.subject[0]):
            raise(ValueError('Specify a subject and a session (-a <SUBJECT> -s <SESSION>).'))

