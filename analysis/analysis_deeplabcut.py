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
        self.file_filter = file_filter
        self.experimenter = os.getlogin()
        
    def parse_arguments(self,arguments = []):
        parser = argparse.ArgumentParser(
            description = 'Animal pose analysis.',
            usage = 'deeplabcut -a <subject> -s <session> -- create|extract|label|run <PARAMETERS>')
        
        parser.add_argument('action',
                            action='store', type=str)
        parser.add_argument('--label-subject',
                            action='store', default=None, type=str)
        parser.add_argument('--label-session',
                            action='store', default=None, type=str)
        parser.add_argument('-c','--example-config',
                            action='store', default='headfixed_top', type=str)
        parser.add_argument('--start',
                            action='store', default=0, type=float)
        parser.add_argument('--stop',
                            action='store', default=1, type=float)
        parser.add_argument('-f','--video-filter',
                            action='store', default='cam0', type=str)
        parser.add_argument('--video-extension',
                            action='store', default='.avi', type=str)
        parser.add_argument('--experimenter',default=os.getlogin(),type=str)
        parser.add_argument('--extract-mode', action='store', default = 'manual')
        parser.add_argument('--extract-algo', action='store', default = 'kmeans')
        parser.add_argument('--extract-user-feedback', action='store_false', default = True)
        parser.add_argument('--extract-crop', action='store_true' ,default = False)

        args = parser.parse_args(arguments[1:])

        self.labeling_session = args.label_session
        self.labeling_subject = args.label_subject
        self.example_config = args.example_config
        self.video_filter = args.video_filter
        self.video_extension = args.video_extension
        self.experimenter = args.experimenter
        self.extractparams = dict(mode = args.extract_mode,
                                  algo = args.extract_algo,
                                  userfeedback = args.extract_user_feedback,
                                  crop = args.extract_crop)
        self.action = args.action
        if self.action == 'create':
            self._run = self._create_project
        elif self.action == 'extract':
            self._run = self._extract_frames_gui
        elif self.action == 'label':
            self._run == self._manual_annotation
        elif self.action == 'run':
            self._run == self._run_dlc
        else:
            raise(ValueError('Available commands are: create, extract, label, and run.'))
        
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
        return config_path

    def get_video_path(self):
        self.session_folders = self.get_sessions_folders()
        video_files = []
        for session in self.session_folders:
            for d in self.datatypes:
                tmp = glob(pjoin(session,d,'*'+self.video_extension))
                for f in tmp:
                    if self.video_filter in f:
                        video_files.append(f)
        return video_files
    
    def _create_project(self):
        # the config file needs to be read and the path updated every time.
        # That is because it uses global paths..
        configpath = self.get_project_folder()
        if not os.path.exists(os.path.dirname(configpath)):
            os.makedirs(configpath)
        import deeplabcut as dlc
        dlc.create_new_project(self.subject[0], self.experimenter, self.get_video_path(),
                               working_directory=configpath,
                               copy_videos=False,
                               multianimal=False)
        
    def _extract_frames_gui(self):
        configpath = self.get_project_folder()
        if not os.path.exists(os.path.dirname(configpath)):
            os.makedirs(configpath)
        import deeplabcut as dlc
        dlc.extract_frames(configpath,
                           **self.extractparams)

    def _open_gui(self):
        # list_stat files
        self.session_folders = self.get_sessions_folders()
        files = []
        for f in self.session_folders:
            files.extend(glob(pjoin(f,self.output_folder,'**','stat.npy'),recursive=True))
        if self.nplanes is None:
            sel = 'combined'
        else:
            sel = 'plane{0}'.format(self.nplanes)
        session = list(filter(lambda x: sel in x,files))
        if len(session):
            print('Opening session: {0}'.format(session[0]))
            if not self.overwrite:
                print('Use the --overwrite to overwrite the results.')
            from suite2p.gui.gui2p import QApplication,MainWindow,warnings,sys

            warnings.filterwarnings("ignore")
            app = QApplication(sys.argv)
            GUI = MainWindow(statfile=session[0])
            ret = app.exec_()

        else:
            print('Could not find session.')
            self.upload = False
            
    def _run(self):

        ops = suite2p.default_ops()
        if not self.nplanes is None:
            ops['nplanes'] = self.nplanes
        if not self.nchannels is None:
            ops['nchannels'] = self.nchannels
        if not self.functional_chan is None:
            ops['functional_channel'] = self.functional_chan
        if not self.tau is None:
            ops['tau'] = self.tau
        if not self.fs is None:
            ops['fs'] = self.fs
        if not self.nonrigid is None:
            ops['nonrigid'] = self.nonrigid
        self.session_folders = self.get_sessions_folders()
        if len(self.session_folders)>1:
            self.isconcatenated = True
            print('''Multisession merge is not implemented.
            Using the first session.''')
        print(self.session_folders)
        db = dict(data_path = [pjoin(f,self.input_folder) for f in self.session_folders],
                  save_path0 = self.session_folders[0])
        # Search for SBX files
        files = []
        for f in db['data_path']:
            files.extend(glob(pjoin(f,'**','*.sbx'),recursive=True))
        files = list(filter(lambda x: self.file_filter in x,files))
        if len(files):
            db['tiff_list'] = files
            ops['input_format'] = 'sbx'
            from sbxreader import sbx_get_metadata
            sbxmeta = sbx_get_metadata(files[0])
            ops['fs'] = sbxmeta['frame_rate']
            ops['nplanes'] = sbxmeta['num_planes']
            if sbxmeta['num_planes']>1:
                ops['plane_depths'] = [d for d in np.array(sbxmeta['etl_pos'])+sbxmeta['stage_pos'][-1]]
            else:
                ops['plane_depths'] = sbxmeta['stage_pos'][-1]
            ops['magnification'] = sbxmeta['magnification']
            ops['aspect'] = sbxmeta['um_per_pixel_x']/sbxmeta['um_per_pixel_y']
            ops['um_per_pixel_x'] = sbxmeta['um_per_pixel_x']
            ops['um_per_pixel_y'] = sbxmeta['um_per_pixel_y']
        if not len(files):
            # then it must be tiff
            files = []
            for f in db['data_path']:
                files.extend(glob(pjoin(f,'**','*.tiff'),recursive=True))
                files.extend(glob(pjoin(f,'**','*.tif'),recursive=True))
                files.extend(glob(pjoin(f,'**','*.TIFF'),recursive=True))
                files.extend(glob(pjoin(f,'**','*.TIF'),recursive=True))
            files = list(filter(lambda x: self.file_filter in x,files))
            print(files)
            raise(NotImplementedError('This needs to be tested'))
        
        if not len(files):
            self.upload = False
            raise(OSError('Could not find files to run Suite2p in this session.'))
        suite2p.run_s2p(ops=ops,db=db)
        
    def validate_parameters(self):
        if len(self.subject)>1:
            raise(ValueError('Specify only one subject for Suite2p.'))
        if not len(self.subject[0]):
            raise(ValueError('Specify a subject and a session for Suite2p (-a <SUBJECT> -s <SESSION>).'))

        if len(self.session)>1:
            self.is_multisession = True
            raise(OSError('Segmenting multiple sessions needs to be implemented.'))
        
        # Suite2p parameters are validated before running, to avoid downloading because we need the data files.

        
