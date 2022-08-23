from labdatatools.analysis import *
import argparse

class AnalysisDeeplabcut(BaseAnalysisPlugin):
    def __init__(self,subject,
                 session = None,
                 datatypes = [''],
                 includes = [''],
                 excludes = [''],
                 labeling_subject = None,
                 labeling_session = None,
                 bwlimit = None,
                 overwrite = False,
                 camera = 'cam0',
                 input_folder = None,
                 file_filter = None,
                 **kwargs):
        super(AnalysisDeeplabcut,self).__init__(
            subject = subject,
            session = session,
            datatypes = datatypes,
            includes = includes,
            excludes = excludes,
            bwlimit = bwlimit,
            overwrite = overwrite,
            **kwargs)
        self.name = 'Deeplabcut'
        self.labeling_folder = 'dlc_labeling'
        self.analysis_folder = 'dlc_analysis'

        self.output_folder = self.labeling_folder
        if labeling_subject is None:
            labeling_subject = subject
        if labeling_subject is None:
            labeling_session = session[0]        
        self.labeling_session = labeling_session
        self.labeling_subject = labeling_subject
        self.video_extension = '.avi'
        self.video_filter = 'cam0'
        self.experimenter = os.getlogin()
        
    def parse_arguments(self,arguments = []):
        parser = argparse.ArgumentParser(
            description = '''
Animal pose analysis.
Actions are: create, extract, label, train, run
''',
            usage = 'deeplabcut -a <subject> -s <session> -- create|extract|label|train|run <PARAMETERS>')
        
        parser.add_argument('action',
                            action='store', type=str, help = "action to perform (CREATE project, EXTRACT frames, manual LABEL frames, TRAIN the network, RUN the analysis on a dataset)")
        parser.add_argument('--label-subject',
                            action='store', default=None, type=str, help = "specity subject used for initial labeling (used when analyzing new videos)")
        parser.add_argument('--label-session',
                            action='store', default=None, type=str, help = "specify session used for initial labeling (used when analyzing new videos)")
        parser.add_argument('-c','--example-config',
                            action='store', default='headfixed_side', type=str)
        parser.add_argument('--start',
                            action='store', default=0, type=float, help = "specify start frame for extracting outlier frames (not implemented yet)")
        parser.add_argument('--stop',
                            action='store', default=1, type=float, help = "specify stop frame for extracting outlier frames (not implemented yet)")
        parser.add_argument('-f','--video-filter',
                            action='store', default='cam0',
                            type=str,
                            help = "indicate which video to load: cam0 (default) for lateral view and cam1 for bottom view")
        parser.add_argument('--video-extension',
                            action='store', default='.avi', type=str, help = "specify video extension, default is .avi")
        parser.add_argument('--experimenter',default=os.getlogin(),type=str, help = "add experimenter as well as which view is being used for this project (lateral or bottom, i.e. GRB-lateral)")
        parser.add_argument('--extract-mode', action='store', default = 'manual', help = "specify if extraction ocurs manual (default) or automatic")
        parser.add_argument('--extract-algo', action='store', default = 'kmeans', help = "if extract-mode = automatic, specify the algorithm to use (uniform or kmeans (default)")
        parser.add_argument('--extract-nouser-feedback', action='store_false',
                            default = True,
                            help="Use user feedback for extraction (default FEEDBACK)")
        parser.add_argument('--extract-crop', action='store_true' ,default = False, help = "specify if user wants to crop video before extracting frames (default is False)")

        args = parser.parse_args(arguments[1:])

        self.labeling_session = args.label_session
        self.labeling_subject = args.label_subject
        self.example_config = args.example_config

        self.video_filter = args.video_filter
        self.video_extension = args.video_extension
        self.experimenter = args.experimenter
        
        self.extractparams = dict(mode = args.extract_mode,
                                  algo = args.extract_algo,
                                  userfeedback = args.extract_nouser_feedback,
                                  crop = args.extract_crop)
        self.action = args.action
        print(self.action)
        if self.action == 'create':
            self._run = self._create_project
        elif self.action == 'extract':
            self._run = self._extract_frames_gui
        elif self.action == 'label':
            self._run = self._manual_annotation
        elif self.action == 'train':
            self._run = self._train_dlc
        elif self.action == 'run':
            self._run = self._run_dlc
        else:
            raise(ValueError('Available commands are: create, extract, label, and run.'))
            
    def get_analysis_folder(self):
        self.session_folders = self.get_sessions_folders()
        session = self.session[0]
        subject = self.subject[0]
        session_key = dict(datapath = self.prefs['paths'][0],
                           subject = subject,
                           session = session)
        path = pjoin(session_key['datapath'],
                     session_key['subject'],
                     session_key['session'],
                     self.analysis_folder)
        return path
    
    def get_project_folder(self):
        self.session_folders = self.get_sessions_folders()
        if self.labeling_session is None:
            self.labeling_session = self.session[0]
        if self.labeling_subject is None:
            self.labeling_subject = self.subject[0]
        session_key = dict(datapath = self.prefs['paths'][0],
                           subject = self.labeling_subject,
                           session = self.labeling_session)
        config_path = pjoin(session_key['datapath'],
                            session_key['subject'],
                            session_key['session'],
                            self.labeling_folder)
        if os.path.exists(config_path):
            # try to get it from the cloud.
            if not self.partial_run in ['run', 'process']:
                rclone_get_data(subject = self.labeling_subject,
                                session = self.labeling_session,
                                datatype = self.labeling_folder)                
        # search for files
        if os.path.exists(config_path):
            folders = glob(pjoin(config_path,'*'))
            if len(folders):
                folders = list(filter(os.path.isdir,folders)) 
                folders = list(filter(lambda x: self.experimenter in x,folders))
            if len(folders):
                config_path = pjoin(config_path,folders[0],'config.yaml')
        return config_path

    def get_video_path(self):
        self.session_folders = self.get_sessions_folders()
        video_files = []
        for session in self.session_folders:
            for d in self.datatypes:
                tmp = glob(pjoin(session,d,'*'+self.video_extension))
                for f in tmp:
                    if not self.video_filter is None:
                        if self.video_filter in f:
                            video_files.append(f)
                    else:
                        video_files.append(f)
        return video_files
    
    def _create_project(self):
        # the config file needs to be read and the path updated every time.
        # That is because it uses global paths..
        configpath = self.get_project_folder()
        if not os.path.exists(os.path.dirname(configpath)):
            os.makedirs(os.path.dirname(configpath))
            from datetime import datetime
            date = datetime.today().strftime('%Y-%m-%d')
            os.makedirs(pjoin(os.path.dirname(configpath),
                              '{0}-{1}-{2}'.format(self.subject[0],
                                                   self.experimenter,
                                                   date),'videos')) # because dlc is psychotic
        if not 'config.yaml' in configpath:
            import deeplabcut as dlc
            dlc.create_new_project(self.subject[0], self.experimenter, self.get_video_path(),
                                   working_directory=configpath,
                                   copy_videos=False,
                                   multianimal=False)
        print('Project already exists ? [{0}]'.format(configpath))
    def _extract_frames_gui(self):
        configpath = self.get_project_folder()
        if not os.path.exists(configpath):
            print('No project found, create it first.')
        import deeplabcut as dlc
        dlc.extract_frames(configpath,
                           **self.extractparams)
        self.overwrite = True

    def _manual_annotation(self):
        configpath = self.get_project_folder()
        if not os.path.exists(configpath):
            print('No project found, create it first.')
        import deeplabcut as dlc
        dlc.label_frames(configpath)
        self.overwrite = True
                    
    def _train_dlc(self):
        configpath = self.get_project_folder()
        if not os.path.exists(configpath):
            print('No project found, create it first.')
        import deeplabcut as dlc
        dlc.create_training_dataset(configpath,
                                    net_type = 'resnet_50',
                                    augmenter_type='imgaug')
        dlc.train_network(configpath,
                          shuffle=1,
                          trainingsetindex=0,
                          gputouse=None,
                          max_snapshots_to_keep=5,
                          autotune=False,
                          displayiters=100,
                          saveiters=15000,
                          maxiters=300000,
                          allow_growth=True)

    def _run_dlc(self):
        configpath = self.get_project_folder()
        if not os.path.exists(configpath):
            print('No project found, create it first.')
        video_files = self.get_video_path()
        if not len(video_files):
            print('No video files found.')
            return
        resfolder = self.get_analysis_folder()
        import deeplabcut as dlc
        dlc.analyze_videos(configpath, video_files,
                           videotype=self.video_extension,
                           shuffle=1,
                           trainingsetindex=0,
                           save_as_csv=True,
                           destfolder=resfolder,
                           dynamic=(True, .5, 10))

    def validate_parameters(self):
        if len(self.subject)>1:
            raise(ValueError('Specify only one subject for DLC.'))
        if not len(self.subject[0]):
            raise(ValueError('Specify a subject and a session for DLC (-a <SUBJECT> -s <SESSION>).'))
        if self.session == ['']:
            raise(ValueError('No session specified.'))
        if len(self.session)>1:
            self.is_multisession = True
            raise(OSError('Segmenting multiple sessions still has to be implemented.'))
        if self.datatypes == ['']:
            raise(ValueError('No datatype specified.'))
