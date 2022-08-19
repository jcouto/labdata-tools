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
                 partial_run = None, # 'get', 'run' or 'put'
                 **kwargs):

        self.description = 'Not Implemented'
        self.name = ''
        self.partial_run = partial_run
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
        if self.partial_run in ['get','fetch']:
            self.fetch_data()
            return
        elif self.partial_run in ['run','process']:
            self._run()
            return
        elif self.partial_run in ['put','upload']:
            self.put_data()
            return
        self.fetch_data()
        self._run()
        self.put_data()

    def slurm(self, analysisargs,
              conda_environment = None,
              ntasks=None,
              ncpuspertask = None,
              memory=None,
              walltime=None,
              partition=None):
        cmd = 'labdata run {0}'.format(self.name.lower())
        if not self.subject == ['']: 
            cmd += ' -a {0}'.format(' '.join(self.subject))
        if not self.session == ['']: 
            cmd += ' -s {0}'.format(' '.join(self.session))
        if not self.datatypes == ['']: 
            cmd += ' -d {0}'.format(' '.join(self.datatypes))
        if not self.includes == []: 
            cmd += ' -i {0}'.format(' '.join(self.includes))
        if not self.excludes ==[]: 
            cmd += ' -e {0}'.format(' '.join(self.excludes))
        if self.overwrite:
            cmd += ' --overwrite'
        if len(analysisargs):
            cmd += ' ' + ' '.join(analysisargs)    
        cmd = self.parse_slurm_cmd(cmd)
        from .slurm import submit_slurm_job
        
        submit_slurm_job(jobname=self.name.lower(),
                         command=cmd,
                         ntasks=ntasks,
                         ncpuspertask = ncpuspertask,
                         memory=memory,
                         walltime=walltime,
                         partition=partition,
                         conda_environment=conda_environment,
                         module_environment=None,
                         mail=None,
                         sbatch_append='')
        
    def parse_slurm_cmd(self, cmd):
        '''Use this to change the command from a subclass.'''
        return cmd
    
    def _run(self):
        raise(NotImplementedError(
            'Use this method to write code to run the analysis.'))

    def parse_arguments(self,arguments):
        'Options go here'
        pass

    def validate_parameters(self):
        'This method is ran first '
        pass
    
    def fetch_data(self,overwrite = False):
        if not self.subject is None:
            if not self.session is None:
                for subject in self.subject:
                    for session in self.session:
                        for datatype in self.datatypes:
                            print('Getting data for session: {0} {1}'.format(subject,session))
                            rclone_get_data(subject = subject,
                                            session = session,
                                            datatype = datatype,
                                            overwrite = overwrite,
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

        
