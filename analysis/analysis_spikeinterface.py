import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from labdatatools.analysis import *
import argparse

class AnalysisSpikeinterface(BaseAnalysisPlugin):
    def __init__(self,subject,
                 session = None,
                 datatypes = [''],
                 includes = [''],
                 excludes = default_excludes,
                 bwlimit = None,
                 overwrite = False,
                 **kwargs):
        '''
labdatatools wrapper for running spike sorting through SpikeInterface.
Joao Couto - May 2022
        '''
        super(AnalysisSpikeinterface,self).__init__(
            subject = subject,
            session = session,
            datatypes = datatypes,
            includes = includes,
            excludes = excludes,
            bwlimit = bwlimit,
            overwrite = overwrite,
            **kwargs)
        self.name = 'spikeinterface'
        self.output_folder = 'kilosort2.5'
        self.datatypes = ['ephys_*']
        if not datatypes == ['']:
            self.input_folder = datatypes[0]
        else:
            self.input_folder = 'ephys_*'
        self.no_noise = False
        self.no_sorting = False
        self.no_waveforms = False
        self.no_syncs = False

    def parse_arguments(self,arguments = []):
        parser = argparse.ArgumentParser(
            description = 'Analysis of spike data using kilosort version 2.5 through spike interface.',
            usage = 'spikeinterface -- <PARAMETERS>')
        
        parser.add_argument('-p','--probe',
                            action='store', default=None, type = int,
                            help = "Probe number to sort or visualize")
        parser.add_argument('--no-noise',
                            action='store_true', default=False,
                            help = "Skip the background noise computation.")
        parser.add_argument('--no-sorting',
                            action='store_true', default=False,
                            help = "Skip spike sorting")
        parser.add_argument('--waveforms',
                            action='store_true', default=False,
                            help = "Skip the waveform computation.")
        parser.add_argument('--no-syncs',
                            action='store_true', default=False,
                            help = "Skip probe syncs")


        args = parser.parse_args(arguments[1:])
        self.probe = args.probe
        self.no_noise = args.no_noise
        self.no_sorting = args.no_sorting
        self.waveforms = args.waveforms
        self.no_syncs = args.no_syncs

    def _run(self):
        import spikeinterface.full as si
        import spikeinterface.sorters as ss
        
        from spikeinterface.core import (get_global_job_kwargs,
                                         set_global_job_kwargs)
        job_kwargs = dict(n_jobs=4,
                          progress_bar=True)
        set_global_job_kwargs(**job_kwargs)
        
        folders = self.get_sessions_folders()        
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
                    probename = os.path.basename(infolder)
                    outfolder = pjoin(folder,self.output_folder,probename)
                    print('Selected subject {0} session {1} probe {2}'.format(
                        self.subject[0],
                        self.session[0],
                        os.path.basename(infolder)))
                    print(outfolder)
                    
                    # find an ap.bin file to sort and read the meta
                    apinfile = glob(pjoin(infolder,'*.ap.bin'))
                    if not len(apinfile):
                        raise(OSError('File not found in {0}'.format(apinfile)))
                                        
                    
                    if not os.path.exists(outfolder):
                        os.makedirs(outfolder)
                    stream_names, stream_ids = si.get_neo_streams('spikeglx', infolder)
                    stream = stream_names[0]
                
                    raw_rec = si.read_spikeglx(infolder, stream_name=stream, load_sync_channel=False)
                    rec1 = si.highpass_filter(raw_rec, freq_min=400.)
                    bad_channel_ids, channel_labels = si.detect_bad_channels(rec1)
                    rec2 = rec1.remove_channels(bad_channel_ids)
                    rec3 = si.phase_shift(rec2)
                    rec = si.common_reference(rec3, operator="median", reference="global")
                    if self.no_noise:
                        noise_levels = si.get_noise_levels(rec, return_scaled=True)
                        if not os.path.exists(outfolder):
                            os.makedirs(outfolder)
                        np.save(pjoin(outfolder,f'{probename}_channel_noise_levels.npy'),noise_levels)
                    if not self.no_sorting:
                        print(f'Running kilosort on {infolder} and saving in {outfolder}')
                        ss.run_kilosort2_5(recording = rec,
                                           output_folder=outfolder,
                                           docker_image=True)
                    if self.waveforms:
                        print(f'Running waveforms and metrics on {outfolder}') 
                        we = get_waveforms_and_metrics(outfolder,aprecording = rec, apfile=apinfile)
                        try:
                            si.export_report(we, pjoin(outfolder,'si_report'),remove_if_exists=True, format='png')
                        except:
                            pass
                    # Extract the probe sync channel
                    if not self.no_syncs:
                        print(f'Extracting syncs from {outfolder}') 
                        sync = si.read_spikeglx(infolder, stream_name=stream, load_sync_channel=True)
                        # extracting the sync channel
                        tt = np.array(sync.get_traces(channel_ids=[sync.get_channel_ids()[-1]])).flatten()
                        sync_onsets,sync_offsets = unpack_npix_sync(tt)
                        syncfile = pjoin(outfolder,f'{probename}_syncs.h5')
                        save_syncs(syncfile,sync_onsets,sync_offsets)
                    
                    if not os.path.exists(outfolder):
                        self.upload = False
    def validate_parameters(self):
        if len(self.subject)>1:
            raise(ValueError('Specify only one subject.'))
        if not len(self.subject[0]) or not len(self.session):
            raise(ValueError('Specify a subject and a session (-a <SUBJECT> -s <SESSION>).'))


def unpackbits(x,num_bits = 16):
    '''
    unpacks numbers in bits.
    '''
    xshape = list(x.shape)
    x = x.reshape([-1,1])
    to_and = 2**np.arange(num_bits).reshape([1,num_bits])
    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])

def unpack_npix_sync(syncdat,srate=1,output_binary = False):
    '''Unpacks neuropixels phase external input data
events = unpack_npix3a_sync(trigger_data_channel)    
    Inputs:
        syncdat               : trigger data channel to unpack (pass the last channel of the memory mapped file)
        srate (1)             : sampling rate of the data; to convert to time - meta['imSampRate']
        output_binary (False) : outputs the unpacked signal
    Outputs
        events        : dictionary of events. the keys are the channel number, the items the sample times of the events.
    Joao Couto - April 2019
    Usage:
Load and get trigger times in seconds:
    dat,meta = load_spikeglx('test3a.imec.lf.bin')
    srate = meta['imSampRate']
    onsets,offsets = unpack_npix_sync(dat[:,-1],srate);
Plot events:
    plt.figure(figsize = [10,4])
    for ichan,times in onsets.items():
        plt.vlines(times,ichan,ichan+.8,linewidth = 0.5)
    plt.ylabel('Sync channel number'); plt.xlabel('time (s)')
    '''
    dd = unpackbits(syncdat.flatten(),16)
    mult = 1
    if output_binary:
        return dd
    sync_idx_onset = np.where(mult*np.diff(dd,axis = 0)>0)
    sync_idx_offset = np.where(mult*np.diff(dd,axis = 0)<0)
    onsets = {}
    offsets = {}
    for ichan in np.unique(sync_idx_onset[1]):
        onsets[ichan] = sync_idx_onset[0][
            sync_idx_onset[1] == ichan]/srate
    for ichan in np.unique(sync_idx_offset[1]):
        offsets[ichan] = sync_idx_offset[0][
            sync_idx_offset[1] == ichan]/srate
    return onsets,offsets

def get_waveforms_and_metrics(sortfolder, 
                              apfile = None,
                              aprecording = None,
                              sparse=True,
                              max_spikes_per_unit=500,
                              ms_before=1.5,ms_after=2.,
                              **job_kwargs):

    import spikeinterface.full as si
    from spikeinterface.core import (get_global_job_kwargs,
                                     set_global_job_kwargs)
    job_kwargs = dict(n_jobs=4,
                      progress_bar=True)
    set_global_job_kwargs(**job_kwargs)
    
    if aprecording is None:
        if apfile is None:
            raise(OSError('Need to supply a path to an AP file'))
        raw_rec = si.read_spikeglx(os.path.dirname(apfile),stream_name = stream, load_sync_channel=False)
        rec1 = si.highpass_filter(raw_rec, freq_min=400.)
        bad_channel_ids, channel_labels = si.detect_bad_channels(rec1)
        rec2 = rec1.remove_channels(bad_channel_ids)
        rec3 = si.phase_shift(rec2)
        rec = si.common_reference(rec3, operator="median", reference="global")
    else:
        rec = aprecording
    sortresults = si.read_sorter_folder(sortfolder)
    sortresults = sortresults.remove_empty_units()
    if not os.path.exists(pjoin(sortfolder,'waveforms')):
        we = si.extract_waveforms(rec, sortresults, folder=pjoin(sortfolder,'waveforms'),
                                  mode="folder",num_spikes_for_sparsity=100, method="radius", radius_um=150,
                                  sparse=sparse, max_spikes_per_unit = max_spikes_per_unit, 
                                  ms_before=ms_before, ms_after=ms_after,precompute_template=('median',),
                                  **job_kwargs)
        _ = si.compute_noise_levels(we)
        _ = si.compute_correlograms(we)
        _ = si.compute_unit_locations(we)
        _ = si.compute_spike_amplitudes(we, **job_kwargs)
        _ = si.compute_template_similarity(we)
        _ = si.compute_spike_locations(we,**job_kwargs)
        metrics = si.compute_quality_metrics(we, metric_names=[['num_spikes',
                                                                'firing_rate',
                                                                'snr',
                                                                'isi_violation',
                                                                'rp_violation',
                                                                'amplitude_median',
                                                                'amplitude_cutoff']])
    else:
        we = si.load_waveforms(pjoin(sortfolder,'waveforms'))
    return we

def save_syncs(filename,sync_onsets,sync_offsets):
    import h5py as h5
    with h5.File(filename,'w') as fd:
        group = fd.create_group('onsets')
        for k in sync_onsets.keys():
            group.create_dataset(f"sync{k}",data = sync_onsets[k])
        group = fd.create_group('offsets')
        for k in sync_offsets.keys():
            group.create_dataset(f"sync{k}",data = sync_offsets[k])
