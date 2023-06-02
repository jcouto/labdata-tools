###---The very basic intro---###
# This analysis plugin takes advantage of labdatatools to download raw data to
# be analyzed from the churchland gdrive and to upload the results of the
# analysis. Here's a use example in the terminal:
#
# labdata run caiman -a LO032 -s 20220215_114758 -d miniscope -- -decay_time 0.6
#
# This will download the miniscope folder from the specified subject and session
# and perform the caiman analysis with a decay_time of 0.6.
from labdatatools.utils import *
from labdatatools import BaseAnalysisPlugin
import argparse
import json

# config_filename = 'caiman_default_params.json'

# defaults_caiman = dict(
#     downsampling_params = dict(
#         scaling_factor = 0.5
#         ),
#     motion_correction_params = dict(
#         fr = 30,                          # movie frame rate
#         decay_time = 0.6,  #length of a typical transient in seconds
#         pw_rigid = False, # flag for pw-rigid motion correction
#         gSig_filt = (3, 3), # size of filter, in general gSig (see below), change this one if algorithm does not work
#         max_shifts = (5, 5), # maximum allowed rigid shift
#         strides = (48, 48), # start a new patch for pw-rigid motion correction every x pixels
#         overlaps = (24, 24), # overlap between pathes (size of patch strides+overlap) maximum deviation allowed for patch with respect to rigid shifts
#         max_deviation_rigid = 3,
#         border_nan = 'copy', #Assignment of border pixel
#         ),
#     cnmfe_params = dict(
#         method_init = 'corr_pnr',
#         p = 1,               # order of the autoregressive system
#         K = None,            # upper bound on number of components per patch, in general None for 1p data
#         gSig = (3, 3),       # gaussian width of a 2D gaussian kernel, which approximates a neuron
#         gSiz = (13, 13),     # average diameter of a neuron, in general 4*gSig+1
#         Ain = None,
#         merge_thr = 0.7,      # merging threshold, max correlation allowed
#         rf = 80,             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
#         stride = 20,    # amount of overlap between the patches in pixels
#         tsub = 2,            # downsampling factor in time for initialization, increase if you have memory problems
#         ssub = 1,            # downsampling factor in space for initialization, increase if you have memory problems
#         low_rank_background = None,  # None leaves background of each patch intact,True performs global low-rank approximation if gnb>0
#         nb = 0,             # number of background components (rank) if positive,
#         nb_patch = 0,        # number of background components (rank) per patch if gnb>0,
#         min_corr = 0.7,       # min peak value from correlation image
#         min_pnr = 8,        # min peak to noise ration from PNR image
#         ssub_B = 2,          # additional downsampling factor in space for background
#         ring_size_factor = 1.4, # radius of ring is gSiz*ring_size_factor
#         only_init = True,    # set it to True to run CNMF-E
#         update_background_components = True,  # sometimes setting to False improve the results
#         normalize_init = False,               # just leave as is
#         center_psf = True,                    # leave as is for 1 photon
#         del_duplicates = True                # whether to remove duplicates from initialization
#         )
#     )
# def caiman_save_params(params = defaults_caiman, fname = config_filename ):
#     if os.path.dirname(config_filename) == '':
#         caiman_prefpath = pjoin(labdata_preferences['plugins_folder'],'caiman',config_filename)
#     else:
#         caiman_prefpath = fname
#     if not os.path.exists(os.path.dirname(caiman_prefpath)):
#         os.makedirs(os.path.dirname(caiman_prefpath))
#     with open(caiman_prefpath,'w') as fd:
#         json.dump(params,fd,indent=True)


# def caiman_load_params(fname = config_filename):
#     if os.path.dirname(config_filename) == '':
#         caiman_prefpath = pjoin(labdata_preferences['plugins_folder'],'caiman',config_filename)
#     else:
#         caiman_prefpath = fname
#     if not os.path.exists(caiman_prefpath):
#         print('Parameters not found, creating one from defaults ({0}).'.format(caiman_prefpath))
#         caiman_save_params(fname = caiman_prefpath)
#     with open(caiman_prefpath,'r') as fd:
#         params = json.load(fd)
#     return params


class AnalysisCaiman(BaseAnalysisPlugin):

    #Ain = None # possibility to seed with predetermined binary masks, if known, it is the initial estimate of spatial filters
    #stride_cnmf = 20 # amount of overlap between the patches in pixels
    #gnb = 0

    def __init__(self,subject,
                 session = None,
                 datatypes = [''],
                 includes = [''],
                 excludes = [''],
                 bwlimit = None,
                 overwrite = False,
                 #round_num = 'second',
                 **kwargs):

        #Inherit methods and attributes from the BaseAnalysisPlugin
        super(AnalysisCaiman,self).__init__(
            subject = subject,
            session = session,
            datatypes = datatypes,
            includes = includes,
            excludes = excludes,
            bwlimit = bwlimit,
            overwrite = overwrite,
            **kwargs)
        self.name = 'caiman'
        self.params =  None

    def parse_arguments(self, arguments = []): #This is currently not implemented for local executions!
        '''Create parser object to be able to easily execute the analysis in the command line later.'''
        parser = argparse.ArgumentParser(
                description = '''
                    Identification of individual neurons in calium imaging data.
                    Actions are: create,  downsample_videos, motion_correction, spatiotemporal_correlation, run_cnmfe
                    ''',
                usage = 'caiman -a <subject> -s <session> -- create|downsample_videos|motion_correction|spatiotemporal_correlation|run_cnmfe <PARAMETERS>')

        parser.add_argument('action',
                            action='store', type=str, help = "input action to perform (MOTION_CORRECT runs the motion correction step; CNMFE runs the cnmfe fitting)")
        # parser.add_argument('-p','--params',
        #                     action='store', type=str, help = "parameter file or path (default is "+config_filename +")",
        #                     default = config_filename)
        parser.add_argument('--n_processes',
                            action='store', type=int, help = "number of parallel processess for motion correction and cnmfe",
                            default = 2)

        args = parser.parse_args(arguments[1:])

        self.action = args.action
        print(self.action)
        # self.params = caiman_load_params(args.params)
        self.n_processes = args.n_processes

        if self.action == 'create': #Method to generate a parameter file
            self._run = self._run_create
        elif self.action == 'downsample_videos':
            self._run = self._run_downsampling
        elif self.action == 'motion_correction':
            self._run = self._run_motion_correction
        elif self.action == 'spatiotemporal_correlation':
            self._run = self._run_spatiotemporal_correlation
        elif self.action == 'run_cnmfe':
            self._run = self._run_cnmfe
        else:
            raise(ValueError('Available command are: create, downsample_videos, motion_correction, spatiotemporal_correlation, run_cnmfe.'))
#---------------------------------------------------------------------------
    
    def load_caiman_params(self):
        '''Load a previously created caiman parameters file'''

        session_folder = self.get_sessions_folders() #Go thorugh the folders
        caiman_params_file = pjoin(session_folder[0], self.name, 'caiman_params.json')
        assert os.path.exists(caiman_params_file), "Make sure to create the caiman_params.json file first by running the create method."
        #Load the params
        with open(caiman_params_file, 'r') as f:
            caiman_params = json.load(f)

        return caiman_params
#-----------------------------------------------------------------------

    def _run_create(self):
        '''Create method to download the data, create the caiman folder and write
        the parameters file'''

        #Define the default paramters, one dict per action, except spatiotemporal correlation
        caiman_params = dict(
            downsampling_params = dict(
                scaling_factor = 0.5 #This will shrink the image to half size along x and y
                ),
            motion_correction_params = dict(
                fr = None,                          # movie frame rate
                decay_time = 0.6,  #length of a typical transient in seconds
                pw_rigid = False, # flag for pw-rigid motion correction
                gSig_filt = (3, 3), # size of filter, in general gSig (see below), change this one if algorithm does not work
                max_shifts = (5, 5), # maximum allowed rigid shift
                strides = (48, 48), # start a new patch for pw-rigid motion correction every x pixels
                overlaps = (24, 24), # overlap between pathes (size of patch strides+overlap) maximum deviation allowed for patch with respect to rigid shifts
                max_deviation_rigid = 3,
                border_nan = 'copy', #Assignment of border pixel
                ),
            cnmfe_params = dict(
                method_init = 'corr_pnr',
                p = 1,               # order of the autoregressive system
                K = None,            # upper bound on number of components per patch, in general None for 1p data
                gSig = (3, 3),       # gaussian width of a 2D gaussian kernel, which approximates a neuron
                gSiz = (13, 13),     # average diameter of a neuron, in general 4*gSig+1
                Ain = None,
                merge_thr = 0.7,      # merging threshold, max correlation allowed
                rf = 80,             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
                stride = 20,    # amount of overlap between the patches in pixels
                tsub = 2,            # downsampling factor in time for initialization, increase if you have memory problems
                ssub = 1,            # downsampling factor in space for initialization, increase if you have memory problems
                low_rank_background = None,  # None leaves background of each patch intact,True performs global low-rank approximation if gnb>0
                nb = 0,             # number of background components (rank) if positive,
                nb_patch = 0,        # number of background components (rank) per patch if gnb>0,
                min_corr = 0.7,       # min peak value from correlation image
                min_pnr = 8,        # min peak to noise ration from PNR image
                ssub_B = 2,          # additional downsampling factor in space for background
                ring_size_factor = 1.4, # radius of ring is gSiz*ring_size_factor
                only_init = True,    # set it to True to run CNMF-E
                update_background_components = True,  # sometimes setting to False improve the results
                normalize_init = False,               # just leave as is
                center_psf = True,                    # leave as is for 1 photon
                del_duplicates = True,                # whether to remove duplicates from initialization
                border_pix = 0 #Exclusion border so that cnmfe patches don't go all the way until the edge of the image
                )
            )

        #Extract the real frame rate from the miniscope_metadata
        session_folder = self.get_sessions_folders() #Go thorugh the folders
        miniscope_metadata_file = pjoin(session_folder[0], self.datatypes[0], 'metaData.json')
        with open(miniscope_metadata_file, 'r') as f:
            miniscope_metadata = json.load(f)
        caiman_params['motion_correction_params']['fr'] = miniscope_metadata['frameRate']

        #Save a caiman parameters file
        outputFolder = os.path.join(session_folder[0], self.name) #Set path directory to caiman folder
        if not os.path.exists(outputFolder):
            os.mkdir(outputFolder)
        with open(pjoin(outputFolder, 'caiman_params.json'),'w') as fd:
            json.dump(caiman_params, fd, indent=True)
#--------------------------------------------------------------------------

    def _run_downsampling(self): #UNDER CONSTRUCTION! Spatially downsample .avi file
        '''Spatially bin the movies'''
        #TODO: create a simlink to the original movies if the scaling_factor is 1.

        from time import time # For time logging
        #Possibly import os here
        #import cv2
        #import numpy as np

#        #Input check
        session_folder = self.get_sessions_folders() #Go thorugh the folders
#        caiman_params_file = pjoin(session_folders, self.name, 'caiman_params.json')
#        assert os.path.exists(caiman_params_file), "Make sure to create the caiman_params.json file first by running the create method."
#        #Load the params
#        with open(caiman_params_file, 'r') as f:
#            caiman_params = json.load(f)

        #print(session_folders)
        fnames = sorted(glob(os.path.join(session_folder[0], self.datatypes[0], '*.avi'))) #These are the complete video paths
        outputFolder = pjoin(session_folder[0], self.name) #Set path directory to caiman folder
        if not os.path.exists(outputFolder):
            os.mkdir(outputFolder)

        caiman_params = self.load_caiman_params() #Use the loader function defined above
        scaling_factor = caiman_params['downsampling_params']['scaling_factor']
        cropping_ROI = None #No cropping implemented so far

        preproc_t = time()

        #Define the matlab run file
        resize_runfile_template = '''
        %----------The formating the inputs-----------------------------------
        outputFolder = '{outputFolder}';
        fnames =  split('{fnames}', ',');
        scalingFactor = {scalingFactor};


        %----------The matlab code--------------------------------------------
        for k = 1:length(fnames)
            current_video = fnames{k}; % Make sure to only have to access the cell array elements once, to avoid issues with the braces...
            vidObj = VideoReader(current_video);
            while hasFrame(vidObj)
                video = read(vidObj);
            end

            %     if applyCrop
            %         cropped_video = zeros([cropROI(4) , cropROI(3) , size(video,3)],'uint8');
            %         %Add one to the video size because counting starts at one
            %         for n = 1:size(video,3)
            %             cropped_video(:,:,n) = imcrop(video(:,:,n), cropROI);
            %         end
            %     else
            %         cropROI = [1,1,size(video,2), size(video,1)];
            %         cropped_video = video;
            %     end

        binned_video = imresize(video,scalingFactor);

        [~, video_name] = fileparts(current_video);
        vidWriter = VideoWriter(fullfile(outputFolder,['binned_' video_name]));
        open(vidWriter)
        framesOut(1:size(binned_video,4)) = struct('cdata',[],'colormap',[]);
        for n = 1:size(binned_video,4)
            framesOut(n).cdata = binned_video(:,:,:,n);
            writeVideo(vidWriter,framesOut(n));
        end
        close(vidWriter);
        clear video; clear cropped_video; clear binned_video;
        display(sprintf('Processed movie no. %d', round(k)))
        end
'''
        resize_runfile = pjoin(outputFolder,'run_miniscope_downsampling_{0}_{1}.m'.format(
            self.subject[0],
            self.session[0],
            ))

        #Create a comma separated string with the video file names that can be split into cells in matlab, getting around the usage of curly braces
        fnames_string = fnames[0]
        for k in range(1,len(fnames)):
            fnames_string = fnames_string + "," + fnames[k]

        with open(resize_runfile,'w') as fd:
            fd.write(resize_runfile_template.format(
                outputFolder = outputFolder,
                fnames = fnames_string,
                scalingFactor = scaling_factor,
                k ='{k}'))
        cmd = """matlab -nodisplay -nosplash -r "run('{0}');exit;" """
        os.system(cmd.format(resize_runfile))
        #Start the down-sampling

#        for vid in fnames:
#            cap = cv2.VideoCapture(vid)
#            frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #Number of frames
#            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
#            frame_dims = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#
#            new_dims = (int(frame_dims[0]*scaling_factor), int(frame_dims[1]*scaling_factor))
#
#            output_file = os.path.join(outputPath, 'binned_' + os.path.splitext(os.path.split(vid)[1])[0] + '.avi')
#
#            wrt = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, new_dims)
#
#            for k in range(frame_number):
#                success, f = cap.read()
#                if not success:
#                    print(f'Something_happend at frame {k}. Check the movie!')
#                frame = np.array(f[:,:,1],) #The video is gray-scale, convert to float to be able to record negative values
#                binned = cv2.resize(frame, new_dims, interpolation = cv2.INTER_AREA)
#                #Rescale the image and round to integer value
#                wrt.write(np.stack((binned, binned, binned), axis=2)) #Reassemble as rgb for saving
#            cap.release()
#            wrt.release()
#            print(f'Processed: {vid}')
#
#        preproc_dict = dict({'scaling_factor': scaling_factor, 'cropping_ROI': cropping_ROI})
#        np.save(os.path.join(outputPath, 'cropping_binning.npy'), preproc_dict)
        print(f'Video binning done in {round(time() - preproc_t)} seconds')
        print('------------------------------------------------')
#-----------------------------------------------------------------------

    def _run_motion_correction(self):
        '''Perform image registration to correct for image shifts due to
        movements of the field of view'''

        import caiman as cm
        #from caiman.source_extraction import cnmf
        from caiman.motion_correction import MotionCorrect
        from caiman.source_extraction.cnmf import params as params

        #import os #For all the file path manipulations
        from time import time # For timne logging
        #import sys #To apppend the python path for the selection GUI
        import numpy as np


        #File Selection:
        session_folder = self.get_sessions_folders() #Go thorugh the folders
        fnames = sorted(glob(pjoin(session_folder[0], self.name, 'binned_*.avi')))

        #Load and initialize parameters
        caiman_params = self.load_caiman_params()
        opts = params.CNMFParams(params_dict = caiman_params['motion_correction_params'])

        #Start a cluster with the specified number of workers
        c, dview, n_processes = cm.cluster.setup_cluster(backend = 'local',
                                                 n_processes = self.n_processes,
                                                 single_thread = False)


#        #File Selection:
#        session_folders = self.get_sessions_folders() #Go thorugh the folders
#        #print(session_folders)
#        fnames = sorted(glob(os.path.join(session_folders[0], self.name, 'binned_*.avi')))
#
#
#
#        # Initial CNMF params:
#        opts = params.CNMFParams(params_dict=self.params['motion_correction_params'])
#        opts.change_params(params_dict= {'fnames': fnames})
        mc_start_time = time()

        # Motion Correction Step:
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)

        fname_mc = mc.fname_tot_els if opts.motion['pw_rigid'] else mc.fname_tot_rig

        if opts.motion['pw_rigid']:
            bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                         np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
        else:
            bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)


        bord_px = 0 if opts.motion['border_nan'] == 'copy' else bord_px
        mc_movie_file = cm.save_memmap(fname_mc, base_name='mc_movie_', order='C',
                                   border_to_0=bord_px) #Saves the memory mappable file to the caiman folder

        mc_end_time = time()

        #Display Elapsed Time:
        print(f"Motion Correction Finished in: {round(mc_end_time - mc_start_time)} s.")

        #Save the Shift Data from Motion Correction:
        rigid_shifts = np.array(mc.shifts_rig) # Retrieve shifts from mc object
        outputPath = pjoin(session_folder[0], self.name) #Set path directory to same location as data
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)

        np.save(pjoin(outputPath, 'motion_correction_shifts'), rigid_shifts) #Save the np array to npy file
        dview.terminate() #Terminate the processes

        #Do a round of clean-up before the uploading the results
        for f in fname_mc:
            os.remove(f) #Delete all the originally generated motion corrected movie files in fortran order

        c_files = glob(pjoin(outputPath, '*.mmap'))
        c_files.remove(mc_movie_file)
        for f in c_files:
            os.remove(f) #Now also delete all the C ordered files that annoyingly also get created with the last call to save_mmap!
#---------------------------------------------------------------------------

    def _run_spatiotemporal_correlation(self):
        '''Calculate median projection, spatiotemporal correlation and peak-to-noise ratio images. These are useful when possibly aligning one session to the other later on'''

        import caiman as cm
        # from caiman.source_extraction import cnmf
        # from caiman.utils.visualization import inspect_correlation_pnr
        from caiman.source_extraction.cnmf import params as params

        # import os #For all the file path manipulations
        from time import time # For time logging
        #import sys #To apppend the python path for the selection GUI
        import numpy as np
        from tqdm import tqdm

        #File Selection:
        session_folder = self.get_sessions_folders() #Go thorugh the folders
        outputPath = pjoin(session_folder[0], self.name) #Set output path directory to same location as data
        
        #Load and initialize parameters
        caiman_params = self.load_caiman_params()
        opts = params.CNMFParams(params_dict = caiman_params['cnmfe_params'])

        # c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
        #                                          n_processes=2,  # number of process to use, changed to 2-cores temporarily cause my pc is hot garbo
        #                                          single_thread=False)

        # #Set up CNMFE Params:
        # opts = params.CNMFParams(params_dict=self.params['cnmfe_params'])

        # #Load Memory Mappable File and Rigid Shifts File:
        # session_folders = self.get_sessions_folders() #Go thorugh the folders
        #print(session_folders)
        #--------Replaced by a border_pix param in the file--------------------
        #rigid_shifts = os.path.join(session_folders[0], self.name, 'motion_correction_shifts.npy' ) #Load rigid shifts file
        #-------------------------------------------------------------
        mc_movie_file = glob(pjoin(session_folders[0], self.name, '*.mmap'))[0] #Load mmap file with order C.
        #print(memmap_file)

        Yr, dims, T = cm.load_memmap(mc_movie_file, mode='r+')
        images = Yr.T.reshape((T,) + dims, order='F')

#        if opts.motion['pw_rigid']:
#            bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
#                                         np.max(np.abs(mc.y_shifts_els)))).astype(np.int)

        # #Use the rigid shifts file to decide on pixels to exclude from border
        # if opts.motion['border_nan'] == 'copy':
        #     bord_px = 0
        # else:
        #     bord_px = np.ceil(np.max(np.abs(rigid_shifts))).astype(np.int)


        #Remove pixels with basically zero intensity but very few
        medProj = np.median(images, axis=0)
        median_bool = medProj < 1
        # for k in range(images.shape[0]):
        #     temp = images[k,:,:]
        #     temp[median_bool] = 0.0001
        #     images[k,:,:] = temp
        (aa,bb) = np.nonzero(median_bool)
        for a,b in tqdm(zip(aa,bb)): images[:,a,b]=0.0001


        #Compute spatiotemporal correlations on images
        corr_image_start = time()
        cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=opts.init['gSig'][0], swap_dim=False) #Computes Peak-to-Noise Ratio
        #Compute the correlation and pnr image on every frame. This takes longer but will yield
        #The actual correlation image that can be used later to align other sessions to this session
        corr_image_end = time()


        print(f"Computed Correlations in: {corr_image_end - corr_image_start} s.")
        
        out_dict = dict({'median_image': medProj,
                         'spatio_temporal_correlation': cn_filter,
                         'peak_to_noise': pnr})
        np.save(pjoin(outputPath, 'summary_images'), out_dict, allow_pickle = True) #Use np.load(...,allow_pickle=True).tolist() to load the file
        # np.save(outputPath + '/spatio_temporal_correlation_image', cn_filter)
        # np.save(outputPath + '/median_projection', medProj)
        # if your images file is too long this computation will take unnecessarily
        # long time and consume a lot of memory. Consider changing images[::1] to
        # images[::5] or something similar to compute on a subset of the data

        # Plot a summary image and set the parameters // Toggled this off to avoid the plotting window opening
        #inspect_correlation_pnr(cn_filter, pnr)

        # Print parameters set above, modify them if necessary based on summary images
        #print(f'The minimum peak correlation is: {opts.init[min_corr]}') # min correlation of peak (from correlation image)
        #print(f'The minimum peak to noise ratio is: {opts.init[min_pnr]}')  # min peak to noise ratio

        # Shuts down parallel pool and restarts
        #dview.terminate()

#-------------------------------------------------------------------------

#TODO: Should the spatiotemporal correlation and the cnmfe be merged together?
    def _run_cnmfe(self): #Seperate spatiotemporal correlation analysis and cnmfe
        '''Run the cnmfe model'''
    
        import caiman as cm
        from caiman.source_extraction import cnmf
        #from caiman.utils.visualization import inspect_correlation_pnr
        from caiman.source_extraction.cnmf import params as params
        import numpy as np

        from time import time # For time logging
        
        #Get the respective session folder
        session_folder = self.get_sessions_folders() #Go thorugh the folders
        outputPath = pjoin(session_folder[0], self.name) #Set output path directory to same location as data
        
        #Load and initialize parameters
        caiman_params = self.load_caiman_params()
        opts = params.CNMFParams(params_dict = caiman_params['cnmfe_params'])
        
        #Load the data
        #rigid_shifts = os.path.join(session_folders[0], self.name, 'motion_correction_shifts.npy' ) #Load rigid shifts file
        mc_movie_file = glob(pjoin(session_folder[0], self.name, '*.mmap'))[0] #Load mmap file with order C.
       
        Yr, dims, T = cm.load_memmap(mc_movie_file, mode='r+')
        images = Yr.T.reshape((T,) + dims, order='F')
        opts.change_params(params_dict = {'dims': dims})
        
        # #Decide what to do with the unassigned border of the image that may
        # #be shifted outside the field of view due to motion.
        # if opts.motion['border_nan'] == 'copy':
        #     bord_px = 0
        # else:
        #     bord_px = np.ceil(np.max(np.abs(rigid_shifts))).astype(np.int)
        
        c, dview, n_processes = cm.cluster.setup_cluster(backend = 'local',
                                                         n_processes = self.n_processes,  
                                                         single_thread = False)
        # #Set up CNMFE Params:
        # opts = params.CNMFParams(params_dict=self.params['cnmfe_params'])

        # session_folders = self.get_sessions_folders() #Go through the folders
        #print(session_folders)
        # rigid_shifts = os.path.join(session_folders[0], self.name, 'rigid_shifts.npy' ) #Load rigid shifts file
        # memmap_file = glob(os.path.join(session_folders[0], self.name, 'memmap_*.mmap'))[0] #Load mmap file with order C.
        # outputPath = os.path.join(session_folders[0], self.name) #Set output path directory to same location as data

        # Yr, dims, T = cm.load_memmap(memmap_file, mode='r+')
        # images = Yr.T.reshape((T,) + dims, order='F')

        # opts.change_params(params_dict = {'dims': dims})


        # if opts.motion['border_nan'] == 'copy':
        #     bord_px = 0
        # else:
        #     bord_px = np.ceil(np.max(np.abs(rigid_shifts))).astype(np.int)



       # Run CNMF-E on Patches:
        cnmfe_start_time = time()

        cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain = self.params['cnmfe_params']['Ain'], params=opts)
#        import ipdb
#        ipdb.set_trace()
        cnm.fit(images)
        cnm.estimates.detrend_df_f() #Detrend the fluorescence traces

        #Display elapsed time for CNMFE step
        cnmfe_end_time = time()
        print(f"Ran Initialization and Fit CNMFE Model in: {round(cnmfe_end_time - cnmfe_start_time)} s.")

        # Save first round of results
        cnm.save(outputPath + '/uncurated_caiman_results.hdf5')
        print("CaImAn Analysis has been Completed.")

        # Shuts down parallel pool and restarts
        dview.terminate()

#-------------------------------------------------------------------------
    
    def curate(self):
        '''Select good neurons and reject noise components and unclear cells'''
        
        import caiman as cm
        from caiman.source_extraction import cnmf
        from cm.source_extraction.cnmf.cnmf import load_CNMF #To load the object with its attributes and methods
        from caiman.source_extraction.cnmf import params as params
        
        #Start with defining the main functions for the selection GUI...
        
        #Function to get the caiman estimates, especially a full size and original dimension A
        def retrieve_caiman_estimates(data_source):
           '''Fetch cnmfe outputs either from a saved h5 file or from a caiman
           object directly.
           
           Parameters
           ----------
           data_source: Name of h5 file or caiman object
           
           Returns
           ------
           
           Usage
           ----------
           A, C, S, F, image_dims, frame_rate, neuron_num, recording_length, movie_file, spMat = load_caiman_estimates(data_source)
           ----------------------------------------------------------------------------
           '''
           
           import numpy as np
           import h5py
           import scipy
           import scipy.sparse
           from pathlib import Path
        
           # Determine the data source - either hdf5 file or caiman object   
           if isinstance(data_source, str): #Loading the data from HDF5    
                hf = h5py.File(data_source, 'r') #'r' for reading ability
        
            # Extract the noisy, extracted and deconvolved calcium signals
            # Use the same variable naming scheme as inside caiman
        
                params = hf.get('params') 
                image_dims = np.array(params['data/dims'])
                frame_rate = np.array(params['data/fr'])
                movie_file = hf.get('mmap_file')[()] # Use [()] notation to access the value of a dataset in h5
                if not isinstance(movie_file, str):
                    movie_file = movie_file.decode() #This is an issue when changing to other operating system
                movie_file = Path(movie_file) #Convert to path
            
                C = np.array(hf.get('estimates/C'))
                S = np.array(hf.get('estimates/S'))
                
                try:
                    F = np.array(hf.get('estimates/F_dff'))
                except:
                    F = None
                
                # Get the sparse matrix with the shapes of the individual neurons
                temp = hf.get('estimates/A')
                
                # Reconstruct the sparse matrix from the provided indices and data
                spMat = scipy.sparse.csc_matrix((np.array(temp['data']),np.array(temp['indices']),
                                    np.array(temp['indptr'])), shape=np.array(temp['shape']))
                
           else: #Directly accessing from caiman object
                image_dims = data_source.dims
                frame_rate = data_source.params.data['fr']
                movie_file = data_source.mmap_file
                movie_file = Path(movie_file) #Convert to path
                
                
                C = data_source.estimates.C
                S = data_source.estimates.S
                if data_source.estimtaes.F_dff is None:
                    F = None
                else:
                    F = data_source.estimtates.F_dff
                
                spMat = data_source.estimates.A
                
            # Retrieve other useful info from the shape of the signal 
           neuron_num = C.shape[0]
           recording_length = C.shape[1]
        
           deMat = np.array(spMat.todense()) # fill the holes and transform to numpy array
        
            # Several important things here: Other than in caiman the output is saved as 
            # n x neuron_num matrix. Therefore, the neuron dimension will be the third one.
            # Also, it is important to set the order of reshaping to 'F' (Fortran).
           A = deMat.reshape(image_dims[0], image_dims[1], neuron_num, order='F')
        
            #---Define the outputs
           print('------------------------')
           print('Successfully loaded caiman data')
           return A, C, S, F, image_dims, frame_rate, neuron_num, recording_length, movie_file, spMat
           #---------------------------------------------------------------
           
    #Define the GUI for the interactive neuron selection
    def neuron_selection_GUI(data_source = None):
        '''run_current_neuron(data_source = None)
        run_current_neuron(data_source = 'C:Users/Documents/analyzedData.hdf5')
        run_current_neuron(data_source = caiman_object)
        This function allows the user to go through individually identified
        putative neurons identified with CNMF-E and select good and discard bad
        ones. Accepted inputs are path string to saved outputs as HDF5 or caiman
        objects directly. When called without arguments the user is able to
        select a saved file. The neurons are presented according to their maximum
        fluorescence intensity value, such that bright components are shown first
        and dimmer "noise" or background components last. Unclassified components
        will be treated as discarded.'''
        #--------------------------------------------------------------------
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider #For interactive control of frame visualization
        from matplotlib.gridspec import GridSpec #To create a custom grid on the figure     
        
        #-------Declare global variables and specify the display parameters 
        global neuron_contour
        global current_neuron #The neuron currently on display
        global accepted_components #List of booleans determining whether to keep a cell or not
        global keep_neuron #List of the indices of the cell that will be kept in the end
            
        current_neuron = 0 #Start with the first neuron
        display_window = 30 # in seconds 
    
        # Load  retrieve the data and load the binary movie file
        A, C, S, F, image_dims, frame_rate, neuron_num, recording_length, movie_file, spatial_spMat = retrieve_caiman_estimates(data_source)
        
        # Load the motion corrected movie (memory mapped)
        Yr = np.memmap(movie_file, mode='r', shape=(image_dims[0] * image_dims[1], recording_length),
                       order='C', dtype=np.float32)
        # IMPORTANT: Pick the C-ordered version of the file and specify the dtype as np.float32 (!!!)
        movie = Yr.T.reshape(recording_length, image_dims[0], image_dims[1], order='F') # Reorder the same way as they do in caiman
        del Yr # No need to keep this one...
            
        #----Initialize the list of booleans for good neurons and a list with the indices    
        accepted_components = [None] * neuron_num # Create a list of neurons you want to keep or reject
        keep_neuron = [None] #Initialize the output of indices of the neurons to refit
        
        #----Sort the cells according to maximum instensity
        intensity_maxima = np.max(C,1) #Get maximum along the second dimension, e.g. within each row
        idx_max_int = np.argsort(-intensity_maxima) #The negative sign make the sorting go in descending order
        
        #Sort the data accordingly
        C = C[idx_max_int,:]
        S = S[idx_max_int,:]
        A = A[:,:,idx_max_int]
        
        #--Function to prepare display range for plotted traces and the frame number according to the given time
        def adjust_display_range(display_time, display_window, frame_rate, recording_length):
            display_frame = int(display_time * frame_rate)
            frame_window = display_window * frame_rate
            frame_range = np.array([np.int(display_frame-frame_window/2), np.int(display_frame+frame_window/2)])
            #Handle the exceptions where the display cannot be symmetric, start and end of the trace
            if frame_range[0] < 0:
                frame_range[0] = 0
                frame_range[1] = frame_window
            elif frame_range[1] > recording_length:
                frame_range[0] = recording_length - display_window * frame_rate
                frame_range[1] = recording_length
                    
            return display_frame, frame_range
                
        #-------Prepare the plot
        fi = plt.figure(figsize=(14,9))
        gs = GridSpec(nrows=4, ncols=4, height_ratios=[6,3.5,0.1,0.4], width_ratios=[5,3,0.5,1.5])
        fi.suptitle(f"Neuron number {current_neuron}", fontsize=18)
        
        movAx = fi.add_subplot(gs[0,0:1]) 
        movAx.xaxis.set_visible(False) #Remove the axes where not needed
        movAx.yaxis.set_visible(False)
        movAx.set_title("Raw movie", fontsize = 14)
        
        maskAx = fi.add_subplot(gs[0,1:4])
        maskAx.xaxis.set_visible(False) #Remove the axes where not needed
        maskAx.yaxis.set_visible(False)
        maskAx.set_title("Individual neuron denoised", fontsize = 14)
        
        traceAx = fi.add_subplot(gs[1,0:2])
        traceAx.set_xlabel('time (s)', fontsize=12)
        traceAx.set_ylabel('Fluorescence intensity (A.U.)', fontsize=12) 
        traceAx.tick_params(axis='x', labelsize=12)
        traceAx.tick_params(axis='y', labelsize=12)
        
        sliBarAx = fi.add_subplot(gs[3,0:2])
        sliBarAx.xaxis.set_visible(False) #Remove the axes where not needed
        sliBarAx.yaxis.set_visible(False)
        
        interactionAx = fi.add_subplot(gs[1,2:4])
        interactionAx.xaxis.set_visible(False) #Remove the axes where not needed
        interactionAx.yaxis.set_visible(False)
        
        #----Start plotting
        #First find the time of peak activity
        display_frame = int(np.where(C[0] == C[0].max())[0])
        # Very annoying transformation of formats:
            #First find the occurence of the maximum as a tuple and access the first
            #element of this tuple, which is an array and needs to be turned into an int!
        display_time = display_frame/frame_rate
        
        #Call the function to prepare display range
        display_frame, frame_range = adjust_display_range(display_time, display_window, frame_rate, recording_length)
        
        #First the corrected movie with contour
        movie_frame = movAx.imshow(movie[display_frame,:,:], cmap='gray', vmin=0, vmax=np.max(movie)) #FIxate the display range here
        neuron_mask = A[:,:,current_neuron] > 0 #Theshold to get binary mask
        neuron_contour = movAx.contour(neuron_mask, linewidths=0.5) #Overlay binary mask on movie frame
        
        #Then plot the de-noised cell activity alone
        pixel_intensity_scaling = 1/np.max(A[:,:,current_neuron])
        max_acti = np.max(C[current_neuron]) #Find the maximum of the denoised trace
        mask_image = maskAx.imshow(A[:,:,current_neuron] * pixel_intensity_scaling * np.abs(C[current_neuron,display_frame]),
                                   cmap='gray', vmin=0, vmax=max_acti)
        #Also take the positive value for A to make sure it is bigge
    
        #Set up the plots for the traces
        time_vect = np.linspace(frame_range[0]/frame_rate, frame_range[1]/frame_rate, np.int(frame_range[1]-frame_range[0])) 
        F_line = traceAx.plot(time_vect, F[current_neuron, frame_range[0]:frame_range[1]], label='Raw fluorescnece trace')
        S_line = traceAx.plot(time_vect, S[current_neuron, frame_range[0]:frame_range[1]], label='Estimated calcium transients')
        vert_line = traceAx.axvline(display_time, color='red')
        traceAx.grid()
        plt.setp(traceAx, xlim=(frame_range[0]/frame_rate, frame_range[1]/frame_rate))
        plt.setp(traceAx, ylim=(-5, round(np.max(F[current_neuron])+5))) #Scale y axis 
        # traceAx.tick_params(axis='x', labelsize=12)
        # traceAx.tick_params(axis='y', labelsize=12)
        # traceAx.xaxis.label.set_size(12)
        # traceAx.yaxis.label.set_size(12)
        traceAx.legend(prop={'size': 12})      
        
        # Now the text display
        # Static
        interactionAx.text(0.05,0.8,'Accept neuron:', fontsize = 12)
        interactionAx.text(0.05,0.7,'Discard neuron:', fontsize = 12)
        interactionAx.text(0.05,0.6,'Forward:', fontsize = 12)
        interactionAx.text(0.05,0.5,'Backward:', fontsize = 12)
        
        interactionAx.text(0.75,0.8,'c', fontsize = 12, fontweight = 'bold')
        interactionAx.text(0.75,0.7,'x', fontsize = 12, fontweight = 'bold')
        interactionAx.text(0.75,0.6,'>', fontsize = 12, fontweight = 'bold')
        interactionAx.text(0.75,0.5,'<', fontsize = 12, fontweight = 'bold')
        
        show_accepted = interactionAx.text(0.5, 0.2, 'Not decided', fontweight = 'bold', fontsize = 12,
            horizontalalignment = 'center', verticalalignment = 'center',
            bbox ={'facecolor':(1,1,1),'alpha':0.9, 'pad':20})
        
        #--------Set up the slider 
        frame_slider = Slider(
            ax=sliBarAx,
            label='Time',
            valmin=0,
            valmax=recording_length/frame_rate, 
            valinit=display_time, 
            valstep=1/frame_rate) #Fix the steps to integers
    
        frame_slider.label.set_size(12)
        frame_slider.vline.set_visible(False)
        
        #--The slider callback
        # The function to be called anytime a slider's value changes
        def frame_slider_update(val):
            
            display_frame, frame_range = adjust_display_range(val, display_window, frame_rate, recording_length)
            movie_frame.set_data(movie[display_frame,:,:])
            
            mask_image.set_data(A[:,:,current_neuron] * pixel_intensity_scaling * np.abs(C[current_neuron,display_frame]))
            
            time_vect = np.linspace(frame_range[0]/frame_rate, frame_range[1]/frame_rate, np.int(frame_range[1]-frame_range[0])) 
            F_line[0].set_xdata(time_vect)
            F_line[0].set_ydata(F[current_neuron, frame_range[0]:frame_range[1]])
            #Stupidly the output of a call to plot is not directly a line object but a list of line
            #objects!
        
            S_line[0].set_xdata(time_vect)
            S_line[0].set_ydata(S[current_neuron, frame_range[0]:frame_range[1]])
            
            #Make the x-axis fit 
            plt.setp(traceAx, xlim=(frame_range[0]/frame_rate, frame_range[1]/frame_rate))
        
            vert_line.set_xdata(np.array([val, val]))
            fi.canvas.draw_idle()
        
        #--Set up the key callback to switch between cells
        #Set the cell number as the parameter to be updated
        def cell_selection_update(event):
            global neuron_contour
            global current_neuron
            global keep_neuron
    
            # It's necessary here to set these to globals so that they can be redefined within the function
            
            if event.key == 'right' or event.key == 'c' or event.key == 'x':
                if event.key == 'c': #The case where we accept the putative neuron
                    accepted_components[current_neuron] = True #This marks an accepted neuron and puts an entry
                elif event.key == 'x':  #The case where we reject the cell
                    accepted_components[current_neuron] = False       
                    
                if current_neuron < neuron_num:
                        current_neuron = current_neuron+1
                        
            elif event.key == 'left':
                if current_neuron > 0: 
                        current_neuron = current_neuron-1
                                
                                
            fi.suptitle(f"Neuron number {current_neuron}", fontsize=18)
            
            #Find the maximum activation of this neuron on the trace and jump to this
            # position
            display_frame = int(np.where(F[current_neuron] == F[current_neuron].max())[0])
            display_time = display_frame/frame_rate
            display_frame, frame_range = adjust_display_range(display_time, display_window, frame_rate, recording_length)
            
            #Adjust frame slider
            frame_slider.set_val(display_time)
            
            #Jump to the respective movie frame
            #movie_frame.set_data(movie[display_frame,:,:])
            
            #Update the contour on the already displayed frame
            #Need to remove the contours first, unfortunately
            for tp in neuron_contour.collections: 
                tp.remove()
                    
            neuron_mask = A[:,:,current_neuron] > 0
            neuron_contour = movAx.contour(neuron_mask, linewidths=0.5)
    
            #Update the denoised plot
            pixel_intensity_scaling = 1/np.max(A[:,:,current_neuron])
            max_acti = np.max(C[current_neuron]) #Find the maximum of the denoised trace
            mask_image.set_data(A[:,:,current_neuron] * pixel_intensity_scaling * np.abs(C[current_neuron,display_frame]))
            mask_image.set_clim(vmin=0, vmax=max_acti)
                
            #Set the plot with the traces accordingly
            F_line[0].set_ydata(F[current_neuron, frame_range[0]:frame_range[1]])
            S_line[0].set_ydata(S[current_neuron, frame_range[0]:frame_range[1]])
            plt.setp(traceAx, ylim=(-5, round(np.max(F[current_neuron])+5))) #Scale y axis
            
            #Finally also update the slider value
            
            
            #Display whether neuron is accepted or not
            if accepted_components[current_neuron] is None: #Not yet determined
                show_accepted.set_text('Not decided')
                show_accepted.set_color((0,0,0))
                show_accepted.set_bbox({'facecolor':(1,1,1),'alpha':0.9, 'pad':20})
            elif accepted_components[current_neuron] == True: #When accepted
                show_accepted.set_text('Accepted')
                show_accepted.set_color((1,1,1))
                show_accepted.set_bbox({'facecolor':(0.23, 0, 0.3),'alpha':0.9, 'pad':20})
            elif accepted_components[current_neuron] == False:
                show_accepted.set_text('Discarded')
                show_accepted.set_color((1,1,1))
                show_accepted.set_bbox({'facecolor':(0.15, 0.15, 0.15),'alpha':0.9, 'pad':20})
                
            fi.canvas.draw_idle()
            
        #----Action when fiugure is closed 
        def on_figure_closing(event):
             global keep_neuron
            
             index_selection = [i for i, val in enumerate(accepted_components) if val] # Only keep indices of accepted neurons and undecided
             
             original_indices = idx_max_int[index_selection] #Map the sorted data back to the original indices
             original_indices = np.sort(original_indices)
             keep_neuron[0:len(original_indices)] = list(original_indices) #Transform back to list
            
             print(f"Selection completed with {len(keep_neuron)} accepted neurons")
             print('-------------------------------------------------------------')
             
        #----Implement the callbacks
        # register the update function with each slider
        frame_slider.on_changed(frame_slider_update)
    
        # register the key presses
        fi.canvas.mpl_connect('key_press_event', cell_selection_update)
        
        # Detect the closing of the figure
        fi.canvas.mpl_connect('close_event', on_figure_closing)
        
        #Block execution until the figure us closed
        plt.show(block = True)
        
        return keep_neuron, frame_slider
        #-------------------------------------------------------------------------    
    
    #Now process the data   
    session_folder = self.get_sessions_folders() #Go thorugh the folders
    outputPath = pjoin(session_folder[0], self.name) #Set output path directory to same location as data
    caiman_obj = load_CNMF(pjoin(outputPath, 'uncurated_caiman_results.hdf5'))
        
    #Run the GUI
    keep_neuron, frame_slider = neuron_selection_GUI(data_source = caiman_obj)
    #Only retain selected neurons
    caiman_obj.estimates.select_components(idx_components = keep_neuron)
    #Save the results
    caiman_obj.save(outputPath + '/caiman_results.hdf5')
    
#-------------------------------------------------------------------------
    def validate_parameters(self):
        if len(self.subject)>1:
            raise(ValueError('Specify only one subject for CaImAn.'))
        if not len(self.subject[0]):
            raise(ValueError('Specify a subject and a session for CaImAn (-a <SUBJECT> -s <SESSION>).'))
        if len(self.session)>1:
            self.is_multisession = True
            raise(OSError('Segmenting multiple sessions needs to be implemented.'))
        if len(self.datatypes) > 1:
            raise ValueError("There should only be a single datatype -d miniscope")
        if self.datatypes[0] != "miniscope":
            raise ValueError("Wrong datatype is being used. Should be miniscope.")
