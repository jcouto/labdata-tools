from labdatatools import *
import argparse
from glob import glob
from os.path import join as pjoin
import os

class AnalysisKilosort25(BaseAnalysisPlugin):
    def __init__(self,subject,
                 session = None,
                 datatypes = [''],
                 includes = [''],
                 excludes = [''],
                 bwlimit = None,
                 overwrite = False,
                 **kwargs):
        super(AnalysisKilosort25,self).__init__(
            subject = subject,
            session = session,
            datatypes = datatypes,
            includes = includes,
            excludes = excludes,
            bwlimit = bwlimit,
            overwrite = overwrite,
            **kwargs)
        self.name = 'kilosort25'
        self.datatypes = ['ephys_*']
        if not datatypes == ['']:
            self.input_folder = datatypes[0]
        else:
            self.input_folder = 'ephys_*'
        self.output_folder = 'kilosort2.5'
        
    def parse_arguments(self,arguments = []):
        parser = argparse.ArgumentParser(
            description = 'Analysis of spike data using kilosort version 2.5.',
            usage = 'kilosort25 -- <PARAMETERS>')
        
        parser.add_argument('-p','--probe',
                            action='store', default=None, type = int,
                            help = "Probe number to sort or visualize")
        parser.add_argument('--phy',
                            action='store', default=False,
                            help = "Open phy for manual curation")

        args = parser.parse_args(arguments[1:])
        self.probe = args.probe
        self.phy = args.phy
        
        
    def _run(self):
        folders = self.get_sessions_folders()
        
        print(folders)
        for folder in folders:
            f = glob(pjoin(folder,self.input_folder))
            if len(f):
                f = f[0]
                # find out how many probes
                probefolders = natsorted(glob(pjoin(f,'*imec*')))
                if self.probe is None:
                    probes_to_run = list(range(len(probefolders)))
                else:
                    probes_to_run = [self.probe]
                for iprobe in probes_to_run:
                    if not iprobe < len(probefolders):
                        raise(ValueError('Selected a probe that does not exist.'))
                    infolder = probefolders[iprobe]
                    outfolder = pjoin(folder,self.output_folder,os.path.basename(infolder))
                    print('Selected subject {0} session {1} probe {2}'.format(
                        self.subject[0],
                        self.session[0],
                        os.path.basename(infolder)))
                    if self.phy:
                        cmd = 'phy template-gui {0}'.format(outfolder,'params.py')
                    print(outfolder)
                    tmpfolder = pjoin(labdata_preferences['plugins_folder'],'kilosort2.5')
                    if not os.path.exists(tmpfolder):
                        os.makedirs(tmpfolder)
                        print('Created {0}'.format(tmpfolder))
                    # find an ap.bin file to sort and read the meta
                    apinfile = glob(pjoin(infolder,'*.ap.bin'))
                    if not len(apinfile):
                        raise(OSError('File not found in {0}'.format(apinfile)))
                    meta = read_spikeglx_meta(apinfile[0].replace('.bin','.meta'))
                    from scipy.io import savemat
                    chmapdict = dict(
                        chanMap = (meta['channel_idx']+1).reshape([-1,1]).astype('float64'),
                        chanMap0ind = meta['channel_idx'].reshape([-1,1]).astype('float64'),
                        xcoords = meta['coords'][:,0].reshape([-1,1]).astype('float64'),
                        ycoords = meta['coords'][:,1].reshape([-1,1]).astype('float64'),
                        connected = np.ones([len(meta['channel_idx']),1],dtype='float64'),
                        name = apinfile[0])
                    if not os.path.exists(outfolder):
                        os.makedirs(outfolder)
                    channelmappath = pjoin(outfolder,'channelmap.mat')
                    savemat(channelmappath,chmapdict)
                    ksortfile = pjoin(tmpfolder,'kilosort_{0}_{1}_{0}.m'.format(
                        self.subject[0],
                        self.session[0],
                        os.path.basename(infolder)))
                    with open(ksortfile,'w') as fd:
                        fd.write(kilosort_run_file.format(
                            outputfolder = outfolder,
                            inputfolder = infolder,
                            channelmapfile = os.path.basename(channelmappath),
                            nchannels = int(meta['nSavedChans']),
                            srate = float(meta['sRateHz']),
                            i='{i}'))
                    cmd = """matlab -nodisplay -nosplash -r "run('{0}');exit;" """
                    os.system(cmd.format(ksortfile))
                    if not os.path.exists(outfolder):
                        self.upload = False
                    if self.phy:
                        break
    def validate_parameters(self):
        if len(self.subject)>1:
            raise(ValueError('Specify only one subject.'))
        if not len(self.subject[0]) or not len(self.session):
            raise(ValueError('Specify a subject and a session (-a <SUBJECT> -s <SESSION>).'))


# this it the file used to run kilosort; all functions should be in the path
kilosort_run_file = '''
%% you need to change most of the paths in this block
outfolder = '{outputfolder}';
inputfolder = '{inputfolder}';

chanMapFile = '{channelmapfile}';

ops.trange    = [0 Inf]; % time range to sort
ops.NchanTOT  = {nchannels}; % total number of channels in your recording

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ops.chanMap = '{channelmapfile}';
ops.fs = {srate};   % sample rate
ops.fshigh = 300;   % frequency for high pass filtering (150)
ops.Th = [10 4];    % threshold on projections for each per pass 
ops.lam = 10;       %  amplitude penalty (0 not used, 10 average, 50 is a lot) 
ops.AUCsplit = 0.9; % isolation for spliting clusters (max = 1)
ops.minFR = 1/50;   % minimum spike rate (Hz)
ops.momentum = [20 400]; % number of samples to average over when building templates [start end] 
ops.sigmaMask = 30;      % spatial constant in um for computing residual variance of spike
ops.ThPre = 8;           % threshold crossings for pre-clustering (in PCA projection space)
ops.sig = 20;            % spatial scale for datashift kernel
ops.nblocks = 5;         % type of data shifting (0 = none, 1 = rigid, >2 = nonrigid)


%% danger, changing these settings can lead to fatal errors
% options for determining PCs
ops.spkTh           = -6;      % spike threshold in standard deviations (-6)
ops.reorder         = 1;       % whether to reorder batches for drift correction. 
ops.nskip           = 25;  % how many batches to skip for determining spike PCs

ops.GPU            = 1; % has to be 1
% ops.Nfilt        = 1024; % max number of clusters
ops.nfilt_factor   = 4; % max number of clusters per good channel (even temporary ones)
ops.ntbuff         = 64;    % samples of symmetrical buffer for whitening and spike detection
ops.NT             = 64*1024+ ops.ntbuff; % must be multiple of 32 + ntbuff (try decreasing if out of memory). 
ops.whiteningRange = 32; % number of channels to use for whitening each channel
ops.nSkipCov       = 25; % compute whitening matrix from every N-th batch
ops.scaleproc      = 200;   % int16 scaling of whitened data
ops.nPCs           = 3; % how many PCs to project the spikes into
ops.useRAM         = 0; % not yet available

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ops.fproc   = fullfile(outfolder, 'temp_wh.dat'); % proc file on a fast SSD
ops.chanMap = fullfile(outfolder, chanMapFile);
%% this block runs all the steps of the algorithm
fprintf('Looking for data inside %s ', outfolder)
disp('');
% main parameter changes from Kilosort2 to v2.5
ops.sig        = 20;  % spatial smoothness constant for registration
ops.fshigh     = 300; % high-pass more aggresively
ops.nblocks    = 5; % blocks for registration. 0 turns it off, 1 does rigid registration. Replaces "datashift" option. 

% is there a channel map file in this folder?
fs = dir(fullfile(inputfolder, 'chan*.mat'));
if ~isempty(fs)
    ops.chanMap = fullfile(outputfolder, fs(1).name);
end

% find the binary file
fs          = [dir(fullfile(inputfolder, '*ap.bin')) dir(fullfile(inputfolder, '*.dat'))];
ops.fbinary = fullfile(inputfolder, fs(1).name);

% preprocess data to create temp_wh.dat
rez = preprocessDataSub(ops);
rez = datashift2(rez, 1); % last input is for shifting data

% ORDER OF BATCHES IS NOW RANDOM, controlled by random number generator
iseed = 1;                 
rez = learnAndSolve8b(rez, iseed); % main tracking and template matching algorithm

% OPTIONAL: remove double-counted spikes - solves issue in which individual spikes are assigned to multiple templates.
% See issue 29: https://github.com/MouseLand/Kilosort/issues/29
rez = remove_ks2_duplicate_spikes(rez);

rez = find_merges(rez, 1);       % final merges
rez = splitAllClusters(rez, 1);  % final splits by SVD
rez = set_cutoff(rez);           % decide on cutoff
rez.good = get_good_units(rez);  % eliminate widely spread waveforms (likely noise)
fprintf('found %d good units', sum(rez.good>0))
disp('');
fprintf('Saving results to Phy  ')
disp('');
rezToPhy(rez, outfolder);            % write to Phy

% write to matlab
rez.cProj = [];
rez.cProjPC = [];
% final time sorting of spikes, for apps that use st3 directly
[~, isort]   = sortrows(rez.st3);
rez.st3      = rez.st3(isort, :);
% Ensure all GPU arrays are transferred to CPU side before saving to .mat
rez_fields = fieldnames(rez);
for i = 1:numel(rez_fields)
    field_name = rez_fields{i};
    if(isa(rez.(field_name), 'gpuArray'))
        rez.(field_name) = gather(rez.(field_name));
    end
end
% save final results as rez2
fprintf('Saving final results for matlab  ')
disp('');
fname = fullfile(outfolder, 'rez2.mat');
save(fname, 'rez', '-v7.3');
'''

def read_spikeglx_meta(metafile):
    '''
    Read spikeGLX metadata file.
    Joao Couto - 2019
    '''
    with open(metafile,'r') as f:
        meta = {}
        for ln in f.readlines():
            tmp = ln.split('=')
            k,val = tmp
            k = k.strip()
            val = val.strip('\r\n')
            if '~' in k:
                meta[k.strip('~')] = val.strip('(').strip(')').split(')(')
            else:
                try: # is it numeric?
                    meta[k] = float(val)
                except:
                    try:
                        meta[k] = float(val) 
                    except:
                        meta[k] = val
    # Set the sample rate depending on the recording mode
    meta['sRateHz'] = meta[meta['typeThis'][:2]+'SampRate']
    try:
        parse_coords_from_spikeglx_metadata(meta)
    except:
        pass
    return meta

def parse_coords_from_spikeglx_metadata(meta,shanksep = 250):
    '''
    Python version of the channelmap parser from spikeglx files.
    Adapted from the matlab from Jeniffer Colonel

    Joao Couto - 2022
    '''
    if not 'imDatPrb_type' in meta.keys():
        meta['imDatPrb_type'] = 0.0 # 3A/B probe
    probetype = int(meta['imDatPrb_type'])

    imro = np.stack([[int(i) for i in m.split(' ')] for m in meta['imroTbl'][1:]])
    chans = imro[:,0]
    banks = imro[:,1]
    connected = np.stack([[int(i) for i in m.split(':')] for m in meta['snsShankMap'][1:]])[:,3]
    if (probetype <= 1) or (probetype == 1100) or (probetype == 1300):
        # <=1 3A/B probe
        # 1100 UHD probe with one bank
        # 1300 OPTO probe
        electrode_idx = banks*384 + chans
        if probetype == 0:
            nelec = 960;    # per shank
            vert_sep  = 20; # in um
            horz_sep  = 32;
            pos = np.zeros((nelec, 2))
            # staggered
            pos[0::4,0] = horz_sep/2       # sites 0,4,8...
            pos[1::4,0] = (3/2)*horz_sep   # sites 1,5,9...
            pos[2::4,0] = 0;               # sites 2,6,10...
            pos[3::4,0] = horz_sep         # sites 3,7,11...
            pos[:,0] = pos[:,0] + 11          # x offset on the shank
            pos[0::2,1] = np.arange(nelec/2) * vert_sep   # sites 0,2,4...
            pos[1::2,1] = pos[0::2,1]                    # sites 1,3,5...

        elif probetype == 1100:   # HD
            nelec = 384      # per shank
            vert_sep = 6    # in um
            horz_sep = 6
            pos = np.zeros((nelec,2))
            for i in range(7):
                ind = np.arange(i,nelec,8)
                pos[ind,0] = i*horz_sep
                pos[ind,1] = np.floor(ind/8)* vert_sep
        elif probetype == 1300: #OPTO
            nelec = 960;    # per shank
            vert_sep  = 20; # in um
            horz_sep  = 48;
            pos = np.zeros((nelec, 2))
            # staggered
            pos[0:-1:2,0] = 0          # odd sites
            pos[1:-1:2,0] = horz_sep   # even sites
            pos[0:-1:2,1] = np.arange(nelec/2) * vert_sep 
    else:
        raise NotImplementedError('Implement 2.0')
    coords = np.vstack([pos[electrode_idx,0],
                        pos[electrode_idx,1]]).T
    idx = np.arange(len(coords))
    meta['coords'] = coords[connected==1,:]
    meta['channel_idx'] = idx[connected==1]
    return idx,coords,connected
