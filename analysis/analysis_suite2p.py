from labdatatools import BaseAnalysisPlugin
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
        self.nplanes = nplanes
        self.nchannels = nchannels
        self.functional_chan = functional_chan
        self.tau = tau
        self.fs = fs
        self.nonrigid = nonrigid
        self.mesoscope = mesoscope
        self.input_folder = input_folder
        self.output_folder = 'suite2p'
        
    def parse_arguments(self,arguments = []):
        parser = argparse.ArgumentParser(
            description = 'Two-photon imaging registration and segmentation.',
            usage = 'suite2p -- <PARAMETERS>')
        
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

        args = parser.parse_args(arguments[1:])
        
        self.nplanes = args.nplanes
        self.nchannels = args.nchannels
        self.functional_chan = args.functional_chan
        self.tau = args.tau
        self.fs = args.fs
        self.nonrigid = args.nonrigid
        self.mesoscope = args.mesoscope
        self.is_multisession = False
        
    def _run(self):
        import suite2p
        ops = suite2p.default_ops()
        if not self.nplanes is None:
            ops['nplanes'] = self.nplanes
        if not self.nchannels is None:
            ops['nchannels'] = self.nchannels
        if not self.functional_channel is None:
            ops['functional_channel'] = self.functional_chan
        if not self.tau is None:
            ops['tau'] = self.tau
        if not self.fs is None:
            ops['fs'] = self.fs
        if not self.nonrigid is None:
            ops['nonrigid'] = self.nonrigid
            
        if len(self.session_folders)>1:
            print('''Multisession merge is not implemented.
            Using the first session.''')
        db = dict(data_path = [self.session_folders[0]],
                  save_path0 = self.session_name)
        
    def validate_parameters(self):
        if len(self.subject)>1:
            raise(ValueError('Specify only one subject for Suite2p.'))
        if not len(self.subject[0]):
            raise(ValueError('Specify a subject and a session for Suite2p (-a <SUBJECT> -s <SESSION>).'))

        if len(self.session)>1:
            self.is_multisession = True
            raise(OSError('Segmenting multiple sessions needs to be implemented.'))
        
        # Suite2p parameters are validated before running, to avoid downloading because we need the data files.

        
