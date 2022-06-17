from .utils import *
from .rclone import rclone_get_dataset

def BaseAnalysisPlugin():
    def __init__(self, subject,
                 session = None,
                 datatypes = [''],
                 includes = [''],
                 excludes = [''],
                 bwlimit = None,
                 overwrite = False,
                 **kwargs):

        self.description = 'Not Implemented'
        self.name = ''
        self.prefs = labdata_preferences
        self.subject = subject
        self.session = session

        self.bwlimit = bwlimit # For upload
        self.input_folder = ''
        self.output_folder = 'analysis'
        
    def get_session_folder(self):
        self.session_keys = dict(datapath = self.prefs['paths'][0],
                                 subject = self.subject,
                                 session = self.session)
        folders = glob(pjoin(self.session_keys['datapath'],
                             self.session_keys['subject'],
                             self.session_keys['session']))
        if len(folders):
            self.session_folder = folders[0]
            return folders[0]
        raise(OSError('[{0}] Could not find session {1} subject {2}'.format(self.name,
                                                                            self.session,
                                                                            self.subject)))
    
    def process(self,fetch = True, push = True):
        '''Run an analysis locally '''
        self.fetch_data()
        self._run()
        self.put_data()
        
    def _run(self):
        raise(NotImplemented('Use this method to write code to run the analysis.'))
        
    def fetch_data(self):
        if not self.subject is None:
            if not self.session is None:
                for subject in args.subject:
                    for session in args.session:
                        for datatype in args.datatype:
                            rclone_get_data(subject = subject,
                                            session = session,
                                            datatype = datatype,
                                            includes = args.includes,
                                            excludes = args.excludes)                
        self.get_session_folder()

    def put_data(self):
        rclone_upload_data(subject = self.subject,
                           session = self.session,
                           datatype = self.input_folder,
                           path_idx = 0,
                           bwlimit = self.bwlimit,
                           overwrite = self.overwrite,
                           excludes = self.excludes)
    
    def slurm(self,
              memory = None,
              ncpus= None,
              queue = None,
              **kwargs):
        raise(NotImplemented('Run analysis on slurm.'))
        

        
