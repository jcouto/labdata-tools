from labdatatools.analysis import *
import argparse

class AnalysisSuite2p(BaseAnalysisPlugin):
    def __init__(self,subject,
                 session = None,
                 datatypes = [''],
                 includes = [''],
                 excludes = [''],
                 bwlimit = None,
                 overwrite = False,
                 nplanes = None,
                 nchannels = 1,
                 functional_chan = 1,
                 tau = 1.5,
                 fs = None,
                 nonrigid=True,
                 mesoscope = False,
                 input_folder = 'two_photon',
                 file_filter = None,
                 **kwargs):
        super(AnalysisSuite2p,self).__init__(
            subject = subject,
            session = session,
            datatypes = datatypes,
            includes = includes,
            excludes = excludes,
            bwlimit = bwlimit,
            overwrite = overwrite,
            **kwargs)
        self.name = 'Suite2p'
        self.output_folder = 'suite2p'
        self.nplanes = nplanes
        self.nchannels = nchannels
        self.functional_chan = functional_chan
        self.tau = tau
        self.fs = fs
        self.nonrigid = nonrigid
        self.mesoscope = mesoscope
        self.input_folder = input_folder
        self.datatypes = [self.input_folder]  # datatypes not used
        self.file_filter = file_filter
        
    def parse_arguments(self,arguments = []):
        parser = argparse.ArgumentParser(
            description = 'Two-photon imaging registration and segmentation.',
            usage = 'suite2p -a <subject> -s <session> -- <PARAMETERS>')
        
        parser.add_argument('--nplanes',
                            action='store', default=None, type=int)
        parser.add_argument('--nchannels',
                            action='store', default=None, type=int)
        parser.add_argument('--functional-chan',
                            action='store', default=None, type=int)
        parser.add_argument('--tau',
                            action='store', default=1.5, type=float)
        parser.add_argument('--fs',
                            action='store', default=None, type=float)
        parser.add_argument('--nonrigid',action='store_false',default = True)
        parser.add_argument('--mesoscope',action='store_true',default = False)
        parser.add_argument('--file-filter',
                            action='store', default=None, type=str)
        parser.add_argument('--open-result', action='store_true',default = False)

        args = parser.parse_args(arguments[1:])
        
        self.nplanes = args.nplanes
        self.nchannels = args.nchannels
        self.functional_chan = args.functional_chan
        self.tau = args.tau
        self.fs = args.fs
        self.nonrigid = args.nonrigid
        self.mesoscope = args.mesoscope
        self.is_multisession = False
        self.file_filter = args.file_filter
        self.open_result = args.open_result
        if self.open_result:
            self.datatypes = [self.output_folder]
            self._run = self._open_gui

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
        import suite2p
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
        if not self.file_filter is None:
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
            if not self.file_filter is None:
                files = list(filter(lambda x: self.file_filter in x,files))
            if len(files):
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

        
