from .utils import *
from .rclone import rclone_get_data,rclone_upload_data

def load_plugins():
    an = dict(files = glob(pjoin(
        labdata_preferences['plugins_folder'],'analysis_*.py')))
    an['names'] = [os.path.basename(
        f).split('_')[-1].replace('.py','') for f in an['files']]
    analysis = []
    for f,n in zip(an['files'],an['names']):
        analysis.append(dict(file=f,name=n,object = None))
    import sys
    sys.path.append(labdata_preferences['plugins_folder'])
    for f in analysis:
        eval('exec("from {0} import Analysis{1}")'.format(
            os.path.basename(f['file']).replace('.py',''),
            f['name'].capitalize()))
        f['object'] = eval("Analysis{0}".format(f['name'].capitalize()))
    return analysis

class BaseAnalysisPlugin(object):
    def __init__(self, subject,
                 session = [''],
                 datatypes = [''],
                 includes = [''],
                 excludes = [''],
                 bwlimit = None,
                 overwrite = False,
                 upload = True,
                 arguments = [],
                 **kwargs):

        self.description = 'Not Implemented'
        self.name = ''
        self.prefs = labdata_preferences
        self.subject = subject
        self.session = session
        self.datatypes = datatypes
        self.includes = includes
        self.excludes = excludes
        self.overwrite = overwrite
        self.upload = upload
        self.bwlimit = bwlimit # For upload
        self.input_folder = ''
        self.output_folder = 'analysis'
        
    def get_sessions_folders(self):
        self.sessions_folders = []
        self.session_keys = []
        for subject in self.subject:
            for session in self.session: # this is a list
                self.session_keys.append(dict(datapath = self.prefs['paths'][0],
                                              subject = subject,
                                              session = session))
                k = self.session_keys[-1]
                folders = glob(pjoin(k['datapath'],
                                     k['subject'],
                                     k['session']))
                if len(folders):
                    self.sessions_folders.append(folders[0])
                else:
                    raise(OSError('[{0}] Could not find session {1} subject {2}'.format(
                        self.name,
                        self.session,
                        self.subject)))
        return self.sessions_folders
    
    def process(self,fetch = True, push = True):
        '''Run an analysis locally '''
        self.validate_parameters()
        self.fetch_data()
        self._run()
        self.put_data()
        
    def _run(self):
        raise(NotImplemented(
            'Use this method to write code to run the analysis.'))

    def parse_arguments(self,arguments):
        'Options go here'
        pass

    def validate_parameters(self):
        'This method is ran first '
        pass
    
    def fetch_data(self):
        if not self.subject is None:
            if not self.session is None:
                for subject in self.subject:
                    for session in self.session:
                        for datatype in self.datatypes:
                            print('Getting data for session: {0} {1}'.format(subject,session))
                            rclone_get_data(subject = subject,
                                            session = session,
                                            datatype = datatype,
                                            includes = self.includes,
                                            excludes = self.excludes)                
        self.get_sessions_folders()

    def put_data(self):
        if self.upload:
            for subject in self.subject:
                for session in self.session:
                    print('Sending data for session: {0} {1}'.format(subject,session))
                    rclone_upload_data(subject = subject,
                                       session = session,
                                       datatype = self.output_folder,
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
        

        
