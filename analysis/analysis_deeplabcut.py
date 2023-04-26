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
        self.experimenter = None

    def parse_arguments(self,arguments = []):
        parser = argparse.ArgumentParser(
            description = '''
Animal pose analysis.
Actions are: create, template, extract, label, train, evaluate, run, verify, outlier, refine, merge
''',
            usage = 'deeplabcut -a <subject> -s <session> -d <datatype> -- create|template|extract|label|train|evaluate|run|video|verify|outlier|refine|merge <PARAMETERS>')

        parser.add_argument('action',
                            action='store', type=str, help = "action to perform (CREATE project, use config TEMPLATE, EXTRACT frames, manual LABEL frames,\
                            TRAIN the network, EVALUATE the trained network's performance, RUN the analysis on a dataset,\
                             create labeled VIDEO, VERIFY model performance, extract OUTLIER frames, REFINE outlier frames, MERGE datasets for retraining after refining)")
        parser.add_argument('--training-set',
                            action='store', default=0, type=int, help = "specify which training set index to use for training and evaluating the network's performance (default is 0)")
        parser.add_argument('--label-subject',
                            action='store', default=None, type=str, help = "specity subject used for initial labeling (used when analyzing new videos)")
        parser.add_argument('--label-session',
                            action='store', default=None, type=str, help = "specify session used for initial labeling (used when analyzing new videos)")
        parser.add_argument('-c','--example-config',
                            action='store', default='headfixed_side', type=str)
        parser.add_argument('--start',
                            action='store', default=0, type=float, help = "specify start frame for extracting outlier frames (not implemented yet/need to add edit config function)")
        parser.add_argument('--stop',
                            action='store', default=1, type=float, help = "specify stop frame for extracting outlier frames (not implemented yet/need to add edit config function)")
        parser.add_argument('-f','--video-filter',
                            action='store', default='cam0',
                            type=str,
                            help = "indicate which video to load: cam0 (default) for lateral view and cam1 for bottom view")
        parser.add_argument('--video-extension',
                            action='store', default='.avi', type=str, help = "specify video extension, default is .avi")
        parser.add_argument('--data-extension', action='store', default='.h5', type=str, help = "specify the data extension to be used, default is .h5")
        parser.add_argument('--experimenter',default=None,type=str, help = "add experimenter as well as which view is being used for this project (lateral or bottom, i.e. GRB-lateral)")
        parser.add_argument('--extract-mode', action='store', default = 'manual', help = "specify if extraction ocurs manual (default) or automatic")
        parser.add_argument('--extract-algo', action='store', default = 'kmeans', help = "if extract-mode = automatic, specify the algorithm to use (uniform or kmeans (default))")
        parser.add_argument('--extract-no-user-feedback', action='store_false',
                            default = True,
                            help="Use user feedback for extraction (default True)")
        parser.add_argument('--extract-crop', action='store_true' ,default = False, help = "specify if user wants to crop video before extracting frames (default is False)")

        args = parser.parse_args(arguments[1:])

        self.labeling_session = args.label_session
        self.labeling_subject = args.label_subject
        self.example_config = args.example_config
        self.training_set = args.training_set
        self.start = args.start #not implemented yet
        self.stop = args.stop #not implemented yet

        self.video_filter = args.video_filter
        self.video_extension = args.video_extension
        self.data_extension = args.data_extension
        self.experimenter = args.experimenter
        if self.experimenter is None:
            self.experimenter = os.getlogin()
        self.extractparams = dict(mode = args.extract_mode,
                                  algo = args.extract_algo,
                                  userfeedback = args.extract_no_user_feedback,
                                  crop = args.extract_crop)
        self.action = args.action
        
        if self.action == 'create':
            self._run = self._create_project
        elif self.action == 'template':
            self._run = self._use_config_template
        elif self.action == 'extract':
            self._run = self._extract_frames_gui
        # elif self.action == 'add': #not implemented yet
        #     self._run = self._add_new_video
        elif self.action == 'label':
            self._run = self._manual_annotation
        elif self.action == 'train':
            self._run = self._train_dlc
        elif self.action == 'evaluate':
            self._run = self._evaluate_dlc
        elif self.action == 'run':
            self._run = self._run_dlc
        elif self.action == 'video':
            self._run = self._labeled_video
        elif self.action == 'verify':
            self._run = self._verify_dlc
        elif self.action == 'outlier':
            self._run = self._extract_outliers
        elif self.action == 'refine':
            self._run = self._refine_outliers
        elif self.action == 'merge':
            self._run = self._merge_datasets
        else:
            raise(ValueError('Available commands are: create, extract, label, evaluate, run, train, verify, outlier, refine, and merge.'))

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

    def _use_config_template(self):
        configpath = self.get_project_folder()
        if not os.path.exists(configpath):
            print('No project found, create it first.')
        import deeplabcut as dlc
        if self.video_filter == 'cam0':
            lateral_template_GRB = {'colormap': 'summer', 'bodyparts': ['Nosetip', 'Whisker_1', 'Whisker_2', 'Whisker_3', 'Whisker_4',
                'Eye_L', 'Eye_R', 'Eye_Up', 'Eye_Down', 'Jaw', 'Ear', 'Hand_L', 'Hand_R', 'Tongue'], 'dotsize':5, 'start':0.5, 'stop':0.55}
            dlc.auxiliaryfunctions.edit_config(configpath, lateral_template_GRB )
            print('cam0 bodyparts have been added to the config file.')
        elif self.video_filter == 'cam1':
            bottom_template_GRB = {'colormap': 'summer', 'bodyparts': ['Port_L', 'Port_R', 'Nose_TopLeft', 'Nose_TopRight',
            'Nose_BottomLeft', 'Nose_BottomRight', 'Whisker_L', 'Whisker_R', 'MouthEdge_L', 'MouthEdge_R', 'Paw_FrontLeft',
            'Paw_FrontRight', 'Paw_RearLeft', 'Paw_RearRight', 'Tail_Base', 'Tongue'], 'dotsize':5, 'start':0.5, 'stop':0.6}
            dlc.auxiliaryfunctions.edit_config(configpath, bottom_template_GRB)
            print('cam1 bodyparts have been added to the config file.')
        else:
            print('Specify which camera to use (video_filter).')

    def get_project_videos_path(self):
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
                project_videos_path = pjoin(config_path,folders[0], 'videos') 
        return project_videos_path    

    def _extract_frames_gui(self):
        import os
        from pathlib import Path
        configpath = self.get_project_folder()
        if not os.path.exists(configpath):
            print('No project found, create it first.')
        import deeplabcut as dlc
        if self.session is not self.labeling_session:
            future_new_video = Path(self.get_video_path()[0])
            head_tail = os.path.split(future_new_video)
            project_videos_path = self.get_project_videos_path()
            future_new_video_path = Path(pjoin(project_videos_path, head_tail[1])) #as in where the video will eventually end up
            print(future_new_video_path)
            if future_new_video_path.is_file():
                print('Video has already been added to the project. Proceeding with extraction.')
                dlc.extract_frames(configpath, **self.extractparams)
                self.overwrite = True
            else:
                print('Video has not been added to the project.\
                      Adding it to project now.')
                new_video = self.get_video_path()
                dlc.add_new_videos(configpath, new_video, copy_videos=False, coords=None, extract_frames=True)
        else:
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
                          trainingsetindex=self.training_set,
                          gputouse=None,
                          max_snapshots_to_keep=5,
                          autotune=False,
                          displayiters=100,
                          saveiters=15000,
                          maxiters=500000,
                          allow_growth=True)

    def _evaluate_dlc(self):
        configpath = self.get_project_folder()
        if not os.path.exists(configpath):
            print('No project found, create it first.')
        import deeplabcut as dlc
        dlc.evaluate_network(configpath,
                             trainingsetindex=self.training_set,
                             plotting=True)

    def _extract_outliers(self):
        configpath = self.get_project_folder()
        if not os.path.exists(configpath):
            print('No project found, create it first.')
        video_dir = self.get_video_dir()
        data_folder = self.get_data_folder_path()
        if not bool(data_folder):
            print('No data file found for the current session (', self.session[0],'). Please analyze the video first.')
            import sys
            sys.exit()            
        def extract_outlier_frames(
            config,
            videos,
            data_folder,
            videotype="",
            shuffle=1,
            trainingsetindex=0,
            outlieralgorithm="jump",
            comparisonbodyparts="all",
            epsilon=20,
            p_bound=0.01,
            ARdegree=3,
            MAdegree=1,
            alpha=0.01,
            extractionalgorithm="kmeans",
            automatic=False,
            cluster_resizewidth=30,
            cluster_color=False,
            opencv=True,
            savelabeled=False,
            copy_videos=False,
            destfolder=None,
            modelprefix="",
            track_method="",
        ):

            import numpy as np
            import pandas as pd
            from pathlib import Path
            from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal
            from deeplabcut.refine_training_dataset.outlier_frames import ExtractFramesbasedonPreselection


            cfg = auxiliaryfunctions.read_config(config)
            bodyparts = auxiliaryfunctions.intersection_of_body_parts_and_ones_given_by_user(
                cfg, comparisonbodyparts
            )
            if not len(bodyparts):
                raise ValueError("No valid bodyparts were selected.")

            track_method = auxfun_multianimal.get_track_method(cfg, track_method=track_method)

            DLCscorer, DLCscorerlegacy = auxiliaryfunctions.get_scorer_name(
                cfg,
                shuffle,
                trainFraction=cfg["TrainingFraction"][trainingsetindex],
                modelprefix=modelprefix,
            )

            Videos = auxiliaryfunctions.get_list_of_videos(videos, videotype)
            if len(Videos) == 0:
                print("No suitable videos found in", videos)

            for video in Videos:
                if destfolder is None:
                    videofolder = str(Path(video).parents[0])
                    videofolder = destfolder
                
                vname = os.path.splitext(os.path.basename(video))[0]

                try:
                    df, dataname, _, _ = auxiliaryfunctions.load_analyzed_data(
                        data_folder, vname, DLCscorer, track_method=track_method
                    )
                    nframes = len(df)
                    startindex = max([int(np.floor(nframes * cfg["start"])), 0])
                    stopindex = min([int(np.ceil(nframes * cfg["stop"])), nframes])
                    Index = np.arange(stopindex - startindex) + startindex

                    df = df.iloc[Index]
                    mask = df.columns.get_level_values("bodyparts").isin(bodyparts)
                    df_temp = df.loc[:, mask]
                    Indices = []
                    if outlieralgorithm == "uncertain":
                        p = df_temp.xs("likelihood", level="coords", axis=1)
                        ind = df_temp.index[(p < p_bound).any(axis=1)].tolist()
                        Indices.extend(ind)
                    elif outlieralgorithm == "jump":
                        temp_dt = df_temp.diff(axis=0) ** 2
                        temp_dt.drop("likelihood", axis=1, level="coords", inplace=True)
                        sum_ = temp_dt.sum(axis=1, level=1)
                        ind = df_temp.index[(sum_ > epsilon ** 2).any(axis=1)].tolist()
                        Indices.extend(ind)
                    elif outlieralgorithm == "fitting":
                        d, o = compute_deviations(
                            df_temp, dataname, p_bound, alpha, ARdegree, MAdegree
                        )
                        # Some heuristics for extracting frames based on distance:
                        ind = np.flatnonzero(
                            d > epsilon
                        )  # time points with at least average difference of epsilon
                        if (
                            len(ind) < cfg["numframes2pick"] * 2
                            and len(d) > cfg["numframes2pick"] * 2
                        ):  # if too few points qualify, extract the most distant ones.
                            ind = np.argsort(d)[::-1][: cfg["numframes2pick"] * 2]
                        Indices.extend(ind)
                    elif outlieralgorithm == "manual":
                        wd = Path(config).resolve().parents[0]
                        os.chdir(str(wd))
                        from deeplabcut.gui import outlier_frame_extraction_toolbox

                        outlier_frame_extraction_toolbox.show(
                            config,
                            video,
                            shuffle,
                            df,
                            savelabeled,
                            cfg.get("multianimalproject", False),
                        )

                    # Run always except when the outlieralgorithm == manual.
                    if not outlieralgorithm == "manual":
                        Indices = np.sort(list(set(Indices)))  # remove repetitions.
                        print(
                            "Method ",
                            outlieralgorithm,
                            " found ",
                            len(Indices),
                            " putative outlier frames.",
                        )
                        print(
                            "Do you want to proceed with extracting ",
                            cfg["numframes2pick"],
                            " of those?",
                        )
                        if outlieralgorithm == "uncertain" or outlieralgorithm == "jump":
                            print(
                                "If this list is very large, perhaps consider changing the parameters "
                                "(start, stop, p_bound, comparisonbodyparts) or use a different method."
                            )
                        elif outlieralgorithm == "fitting":
                            print(
                                "If this list is very large, perhaps consider changing the parameters "
                                "(start, stop, epsilon, ARdegree, MAdegree, alpha, comparisonbodyparts) "
                                "or use a different method."
                            )

                        if not automatic:
                            askuser = input("yes/no")
                        else:
                            askuser = "Ja"

                        if (
                            askuser == "y"
                            or askuser == "yes"
                            or askuser == "Ja"
                            or askuser == "ha"
                        ):  # multilanguage support :)
                            # Now extract from those Indices!
                            ExtractFramesbasedonPreselection(
                                Indices,
                                extractionalgorithm,
                                df,
                                video,
                                cfg,
                                config,
                                opencv,
                                cluster_resizewidth,
                                cluster_color,
                                savelabeled,
                                copy_videos=copy_videos,
                            ) #there is an error that pops up here mainly due to DLC's add.py function 
                            #being called here. In the code it can be fixed by modifying the function an,d
                            #switching "mklink" to "ln -s"
                        else:
                            print(
                                "Nothing extracted, please change the parameters and start again..."
                            )
                except FileNotFoundError as e:
                    print(e)
                    print(
                        "It seems the video has not been analyzed yet, or the video is not found! "
                        "You can only refine the labels after the a video is analyzed. Please run 'analyze_video' first. "
                        "Or, please double check your video file path"
                    )
    
        extract_outlier_frames(config = configpath, videos = self.get_video_path(), videotype = self.video_extension, data_folder = data_folder[0], trainingsetindex = self.training_set, copy_videos=False) 
        #should call local function now 1/22 -GRB
        #^this is popping up as not defined. maybe I need to define before this line?
        #will investigate 1/25

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
                           trainingsetindex=self.training_set,
                           save_as_csv=True,
                           destfolder=resfolder,
                           dynamic=(True, .5, 10)) #ask Joao why this was set to True 1/25

    def _labeled_video(self):
        configpath = self.get_project_folder()
        if not os.path.exists(configpath):
            print('No project found, create it first.')
        video_path = self.get_video_path()
        resfolder = self.get_analysis_folder()
        import deeplabcut as dlc
        dlc.create_labeled_video(configpath, 
                                video_path, 
                                videotype=self.video_extension, 
                                destfolder=resfolder)

    def _verify_dlc(self):
        configpath = self.get_project_folder()
        if not os.path.exists(configpath):
            print('No project found, create it first.')
        video_path = self.get_video_path()
        data_files = self.get_data_path()
        # print(data_files)
        # import sys
        # sys.exit()
        if not len(video_path):
            print('No video files found.')
            return
        resfolder = self.get_analysis_folder()
        from wfield.io import VideoStack
        import numpy as np
        import pandas as pd
        # import sys
        from vispy import app as vapp
        vapp.use_app()
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
        val = [0]

        # allocate coordinates for all bodyparts for first frame
        x = []
        y = []
        for label in range(nbodyparts):
            x.append(dlc_coords_x[label, int(val[0])])
            y.append(dlc_coords_y[label, int(val[0])])

        frame = mov[int(np.mod(val[0],len(mov)-1))].squeeze()

        # make vispy widget
        fig = vp.Fig(size=(800, 600), show=True)
        plot = fig[0, 0]
        plot.bgcolor = "#efefef"

        # to update coordinates by frame
        def set_data(v):
            x = []
            y = []
            for label in range(nbodyparts):
                x.append(dlc_coords_x[label, int(v)])
                y.append(dlc_coords_y[label, int(v)])
            frame = mov[int(np.mod(v  ,len(mov)-1))].squeeze()
            pl.set_data(np.vstack([x,y]).T)
            im.set_data(frame)
            plot.title.text = str(v)

        # plot and show data on vispy widget
        pl = plot.plot(data=np.vstack([x,y]).T,symbol='o',marker_size=3,width = 0,face_color='k',edge_color='k')
        im = plot.image(frame, cmap="gray")

        plot.camera.set_range((mov.shape[2],0), (mov.shape[3],0)) # flip upside down video
        #cb = plot.colorbar(position="bottom",
        #                   label="frame",
        #                   clim=("0", str(len(mov))),
        #                   cmap="gray",
        #                   border_width=1,
        #                   border_color="#aeaeae")
        #print('Done.')
        # opens slider window
        #app = QApplication(sys.argv)
        #sliderwindow = DLCvideomaker()
        #sliderwindow.show()
        #vapp.run(allow_interactive = True)
        #sliderwindow.grid.addWidget(fig.scene.canvas,0,1)
        @fig.connect
        def on_key_press(event, val = val):
            if event.key == 'Left':
                v = 1
                if 'Shift' in event.modifiers:
                    v = 1000
                val[0] -= v
                set_data(val[0])
            elif event.key == 'Right':
                v = 1
                if 'Shift' in event.modifiers:
                    v = 1000
                val[0] += v
                set_data(val[0])
        fig.show()
        vapp.run()
        #sys.exit(app.exec_())

    def validate_parameters(self):
        if len(self.subject)>1:
            raise(ValueError('Specify only one subject for DLC.'))
        if not len(self.subject[0]):
            raise(ValueError('Specify a subject and a session for DLC (-a <SUBJECT> -s <SESSION>).'))
        if self.session == ['']:
            raise(ValueError('No session specified.'))
        if len(self.session)>1:
            print(self.session)
            self.is_multisession = True
            raise(OSError('Segmenting multiple sessions still has to be implemented.'))
        if self.datatypes == ['']:
            raise(ValueError('No datatype specified.'))

#Modified DLC auxiliaryfunctions
