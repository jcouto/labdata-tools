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
        self.data_extension = '.h5'
        self.video_filter = 'cam0'
        self.experimenter = os.getlogin()

    def parse_arguments(self,arguments = []):
        parser = argparse.ArgumentParser(
            description = '''
Animal pose analysis.
Actions are: create, extract, label, train, run, verify, outlier, refine, merge
''',
            usage = 'deeplabcut -a <subject> -s <session> -d <datatype> -- create|extract|label|train|run|verify|outlier|refine|merge <PARAMETERS>')

        parser.add_argument('action',
                            action='store', type=str, help = "action to perform (CREATE project, EXTRACT frames, manual LABEL frames, TRAIN the network, RUN the analysis on a dataset, \
                            VERIFY model performance, extract OUTLIER frames, REFINE outlier frames, MERGE datasets for retraining after refining)")
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
        parser.add_argument('--data-extension', action='store', default='.h5', type=str, help = "specify the data extension to be used, default is .h5")
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
        self.data_extension = args.data_extension
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
        elif self.action == 'verify':
            self._run = self._verify_dlc
        elif self.action == 'outlier':
            self._run = self._extract_outliers
        elif self.action == 'refine':
            self._run = self._refine_outliers
        elif self.action == 'merge':
            self._run = self._merge_datasets
        else:
            raise(ValueError('Available commands are: create, extract, label, run, train, verify, outlier, refine, and merge.'))

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
        video_path = []
        for session in self.session_folders:
            for d in self.datatypes:
                tmp = glob(pjoin(session,d,'*'+self.video_extension))
                for f in tmp:
                    if not self.video_filter is None:
                        if self.video_filter in f:
                            video_path.append(f)
                    else:
                        video_path.append(f)
        return video_path

    def get_video_dir(self):
        self.session_folders = self.get_sessions_folders()
        video_path = ''
        for session in self.session_folders:
            for d in self.datatypes:
                video_dir = glob(pjoin(session,d))
                # video_dir = ''.join(video_dir)
                # video_path+=video_dir
        # return video_path
        return video_dir

    def get_data_path(self):
        self.session_folders = self.get_sessions_folders()
        data_files = []
        for session in self.session_folders:
            tmp = glob(pjoin(session, self.analysis_folder,'*'+self.data_extension))
            for f in tmp:
                if self.video_filter in f:
                    data_files.append(f)
        return data_files

    def get_data_folder_path(self):
        self.session_folders = self.get_sessions_folders()
        data_folder = []
        for session in self.session_folders:
            data_folder = glob(pjoin(session, self.analysis_folder))
        return data_folder

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

    def _extract_outliers(self):
        configpath = self.get_project_folder()
        if not os.path.exists(configpath):
            print('No project found, create it first.')
        video_dir = self.get_video_dir()
        data_folder = self.get_data_folder_path()
        import deeplabcut as dlc
        dlc.extract_outlier_frames(config = configpath, videos=video_dir, videotype = 'avi', data_folder = data_folder[0])

    def _refine_outliers(self):
        configpath = self.get_project_folder()
        if not os.path.exists(configpath):
            print('No project found, create it first.')
        import deeplabcut as dlc
        dlc.refine_labels(configpath)

    def _merge_datasets(self):
        configpath = self.get_project_folder()
        if not os.path.exists(configpath):
            print('No project found, create it first.')
        import deeplabcut as dlc
        dlc.merge_datasets(configpath)

    def _run_dlc(self):
        configpath = self.get_project_folder()
        if not os.path.exists(configpath):
            print('No project found, create it first.')
        video_path = self.get_video_path()
        if not len(video_path):
            print('No video files found.')
            return
        resfolder = self.get_analysis_folder()
        import deeplabcut as dlc
        dlc.analyze_videos(configpath, video_path,
                           videotype=self.video_extension,
                           shuffle=1,
                           trainingsetindex=0,
                           save_as_csv=True,
                           destfolder=resfolder,
                           dynamic=(True, .5, 10))

    def _verify_dlc(self):
        configpath = self.get_project_folder()
        if not os.path.exists(configpath):
            print('No project found, create it first.')
        video_path = self.get_video_path()
        data_files = self.get_data_path()
        if not len(video_path):
            print('No video files found.')
            return
        resfolder = self.get_analysis_folder()
        from wfield.io import VideoStack
        import numpy as np
        import pandas as pd
        import sys
        from vispy import plot as vp
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import (QApplication, QGridLayout, QGroupBox, QRadioButton, QVBoxLayout, QWidget, QSlider)

        # define the slider window
        class DLCvideomaker(QWidget):
            def __init__(self, parent=None):
                super(DLCvideomaker, self).__init__(parent)

                self.grid = QGridLayout()
                self.grid.addWidget(self.createExampleGroup(), 0, 0)
                self.setLayout(self.grid)

                self.setWindowTitle("DLCvideomaker")
                self.resize(400, 300)

            def createExampleGroup(self):
                groupBox = QGroupBox("Video slider")

                radio1 = QRadioButton("&Frames")

                slider = QSlider(Qt.Horizontal)
                slider.setFocusPolicy(Qt.StrongFocus)
                slider.setTickPosition(QSlider.TicksBothSides)
                slider.setTickInterval(10)
                slider.setMaximum(len(mov)-1)
                slider.setSingleStep(1)
                slider.valueChanged.connect(set_data)
                radio1.setChecked(True)

                vbox = QVBoxLayout()
                vbox.addWidget(radio1)
                vbox.addWidget(slider)
                vbox.addStretch(1)
                groupBox.setLayout(vbox)

                return groupBox

        # load video and data
        mov = VideoStack([video_path[0]], outputdict={'-pix_fmt':'gray'})
        dlc_coords = pd.read_hdf(data_files[0])
        bpts = dlc_coords.columns.get_level_values("bodyparts")
        all_bpts = bpts.values[::3]
        dlc_coords_x, dlc_coords_y, dlc_coords_likelihood = dlc_coords.values.reshape((len(dlc_coords), -1, 3)).T
        bplist = bpts.unique().to_list()
        nbodyparts = len(bplist)
        val = 0

        # allocate coordinates for all bodyparts for first frame
        x = []
        y = []
        for label in range(nbodyparts):
            x.append(dlc_coords_x[label, int(val)])
            y.append(dlc_coords_y[label, int(val)])

        frame = mov[int(np.mod(val,len(mov)-1))].squeeze()

        # make vispy widget
        fig = vp.Fig(size=(800, 600), show=False,vsync=True)
        plot = fig[0, 0]
        plot.bgcolor = "#efefef"

        # to update coordinates by frame
        def set_data(val):
            x = []
            y = []
            for label in range(nbodyparts):
                x.append(dlc_coords_x[label, int(val)])
                y.append(dlc_coords_y[label, int(val)])
            frame = mov[int(np.mod(val,len(mov)-1))].squeeze()
            pl.set_data(np.vstack([x,y]).T)
            im.set_data(frame)

        # plot and show data on vispy widget
        pl = plot.plot(data=np.vstack([x,y]).T,symbol='o',marker_size=3,width = 0,face_color='k',edge_color='k')
        im = plot.image(frame, cmap="gray")
        plot.camera.set_range((mov.shape[2],0), (mov.shape[3],0)) # flip upside down video


        # opens slider window
        app = QApplication(sys.argv)
        sliderwindow = DLCvideomaker()
        sliderwindow.show()
        #sliderwindow.grid.addWidget(fig.scene.canvas,0,1)
        fig.show()

        sys.exit(app.exec_())                           

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
