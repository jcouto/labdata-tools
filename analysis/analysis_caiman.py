###---The very basic intro---###
# This analysis plugin takes advantage of labdatatools to download raw data to
# be analyzed from the churchland gdrive and to upload the results of the
# analysis. Here's a use example in the terminal:
#
# labdata submit caiman -a LO032 -s 20220215_114758 -d miniscope -- run_cnmfe --n_processes 30
#
# This will download the miniscope folder from the specified subject and session
# and perform the caiman analysis with using 30 parallel processes

from labdatatools.utils import *
from labdatatools import BaseAnalysisPlugin
import argparse
import json

class AnalysisCaiman(BaseAnalysisPlugin):

    def __init__(self,subject,
                 session = None,
                 datatypes = [''],
                 includes = [''],
                 excludes = [''],
                 bwlimit = None,
                 overwrite = False,
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
        self.params =  None #These will be the parameters passed via the arguments 

    def parse_arguments(self, arguments = []): #This is currently not implemented for local executions!
        '''Create parser object to be able to easily execute the analysis in the command line later.'''
        parser = argparse.ArgumentParser(
                description = '''
                    Identification of individual neurons in calium imaging data.
                    Actions are: create, caiman_pipeline, curate
                    Note that the -- create method generates a parameters file that will be used for all the analysis steps and that can be edited manually by the user.
                    ''',
                usage = 'caiman -a <subject> -s <session> -d <datatype> -- create|caiman_pipeline|curate <PARAMETERS>')

        parser.add_argument('action',
                            action='store', type=str, help = "Input action to perform, see above.")
        parser.add_argument('--n_processes',
                            action='store', type=int, help = "Number of parallel processess for motion correction and cnmfe",
                            default = 2)

        args = parser.parse_args(arguments[1:])

        self.action = args.action
        print(self.action)
        self.n_processes = args.n_processes

        if self.action == 'create': #Method to generate a parameter file. This has to be run before anything else.
            self._run = self._run_create
        elif self.action == 'caiman_pipeline':
            self._run = self._run_caiman_pipeline
        elif self.action == 'curate':
            self._run = self._run_curate
        else:
            raise(ValueError('Available command are: create, caiman_pipeline, curate.'))
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
        fr = miniscope_metadata['frameRate']
        if isinstance(fr,str): #Sometimes the format comes up as a string of type "20FPS"
            fr = int(fr[:-3]) #Remove "FPS" from string and convert to integer
        caiman_params['motion_correction_params']['fr'] = fr

        #Save a caiman parameters file
        output_folder = os.path.join(session_folder[0], self.name) #Set path directory to caiman folder
        self.output_folder = self.name #Here this means the datatype folder only
        
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        with open(pjoin(output_folder, 'caiman_params.json'),'w') as fd:
            json.dump(caiman_params, fd, indent=True)
#--------------------------------------------------------------------------

    def _run_caiman_pipeline(self): #UNDER CONSTRUCTION! Spatially downsample .avi file
        '''Run the entire caiman analysis. This includes first spatially down-sampling
        the videos if desired, correcting for movement artificats on the movies, calculating
        summary images (spatio-temporal correlation and peak-to-noise ratio) and then
        actually running cnmfe on the data.'''
        #TODO: create a simlink to the original movies if the scaling_factor is 1.

        from time import time # For time logging
        import caiman as cm
        from caiman.motion_correction import MotionCorrect
        from caiman.source_extraction.cnmf import params as params
        from caiman.source_extraction import cnmf
        import numpy as np
        from tqdm import tqdm #For time progress bar
        from natsort import natsorted


        session_folder = self.get_sessions_folders() #Go thorugh the folders
        fnames = natsorted(glob(os.path.join(session_folder[0], self.datatypes[0], '*.avi'))) #These are the complete video paths
        print(f'{fnames}')
        output_folder = pjoin(session_folder[0], self.name) #Set path directory to caiman folder
        self.output_folder = self.name #Here this means the datatype folder only
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

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
        resize_runfile = pjoin(output_folder,'run_miniscope_downsampling_{0}_{1}.m'.format(
            self.subject[0],
            self.session[0],
            ))

        #Create a comma separated string with the video file names that can be split into cells in matlab, getting around the usage of curly braces
        fnames_string = fnames[0]
        for k in range(1,len(fnames)):
            fnames_string = fnames_string + "," + fnames[k]

        with open(resize_runfile,'w') as fd:
            fd.write(resize_runfile_template.format(
                outputFolder = output_folder,
                fnames = fnames_string,
                scalingFactor = scaling_factor,
                k ='{k}'))
        cmd = """matlab -nodisplay -nosplash -r "run('{0}');exit;" """
        os.system(cmd.format(resize_runfile))
        
        print(f'Video binning done in {round(time() - preproc_t)} seconds')
        print('------------------------------------------------')
        
        #-----------------------------------------------------------------------
        #'''Perform image registration to correct for image shifts due to
        #movements of the field of view'''


        ##File Selection:
        #session_folder = self.get_sessions_folders() #Go thorugh the folders
        fnames = natsorted(glob(pjoin(session_folder[0], self.name, 'binned_*.avi')))

        #Retrieve motion correction parameters
        opts = params.CNMFParams(params_dict = caiman_params['motion_correction_params'])

        #Start a cluster with the specified number of workers
        c, dview, n_processes = cm.cluster.setup_cluster(backend = 'local',
                                                 n_processes = self.n_processes,
                                                 single_thread = False)

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
        #output_folder = pjoin(session_folder[0], self.name) #Set path directory to same location as data
        #self.output_folder = self.name #Here this means the datatype folder only
        #if not os.path.exists(output_folder):
        #    os.mkdir(output_folder)

        np.save(pjoin(output_folder, 'motion_correction_shifts'), rigid_shifts) #Save the np array to npy file
        dview.terminate() #Terminate the processes

        #Do a round of clean-up before the uploading the results
        for f in fname_mc:
            os.remove(f) #Delete all the originally generated motion corrected movie files in fortran order

        c_files = glob(pjoin(output_folder, '*.mmap'))
        c_files.remove(mc_movie_file)
        for f in c_files:
            os.remove(f) #Now also delete all the C ordered files that annoyingly also get created with the last call to save_mmap!

        #---------------------------------------------------------------------------
        #'''Calculate median projection, spatiotemporal correlation and peak-to-noise ratio images. These are useful when possibly aligning one session to the other later on'''

        #Set the caiman pramaeters
        opts = params.CNMFParams(params_dict = caiman_params['cnmfe_params'])

        mc_movie_file = glob(pjoin(session_folder[0], self.name, '*.mmap'))[0] #Load mmap file with order C.
        Yr, dims, T = cm.load_memmap(mc_movie_file, mode='r+')
        images = Yr.T.reshape((T,) + dims, order='F')

        #Remove pixels with basically zero intensity but sporadic jumps that may corrupt the peak to noise estimation
        medProj = np.median(images, axis=0)
        median_bool = medProj < 1
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
        np.save(pjoin(output_folder, 'summary_images'), out_dict, allow_pickle = True) #Use np.load(...,allow_pickle=True).tolist() to load the file

#-------------------------------------------------------------------------

        #'''Run the cnmfe model'''

        #Use cnmfe params
        opts = params.CNMFParams(params_dict = caiman_params['cnmfe_params'])
        
        #Load the data (obsolete)
        #mc_movie_file = glob(pjoin(session_folder[0], self.name, '*.mmap'))[0] #Load mmap file with order C.
       
        #Yr, dims, T = cm.load_memmap(mc_movie_file, mode='r+')
        #images = Yr.T.reshape((T,) + dims, order='F')
        opts.change_params(params_dict = {'dims': dims})
        
        c, dview, n_processes = cm.cluster.setup_cluster(backend = 'local',
                                                         n_processes = self.n_processes,  
                                                         single_thread = False)
        
        print(f'Images file name is: {images.filename}')
       # Run CNMF-E on Patches:
        cnmfe_start_time = time()

        cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain = caiman_params['cnmfe_params']['Ain'], params=opts)
        cnm.fit(images)
        print(f'cnm mmap file is: {cnm.mmap_file}')
        cnm.estimates.detrend_df_f() #Detrend the fluorescence traces
        print(f'cnm mmap file stil is: {cnm.mmap_file}')
        #Display elapsed time for CNMFE step
        cnmfe_end_time = time()
        print(f"Ran Initialization and Fit CNMFE Model in: {round(cnmfe_end_time - cnmfe_start_time)} s.")

        # Save first round of results
        cnm.save(output_folder + '/uncurated_caiman_results.hdf5')
        print("CaImAn Analysis has been Completed.")

        # Shuts down parallel pool
        dview.terminate()
        del cnm #This might be helpful because reloading this object when it is already present takes more than 10x times as long
        del Yr
        del images

#-------------------------------------------------------------------------

#-----------------------------------------------------------------------------    
    def _run_curate(self):
        '''Select good neurons and reject noise components and unclear cells'''
        
        import caiman as cm
        from caiman.source_extraction import cnmf
        from caiman.source_extraction.cnmf.cnmf import load_CNMF #To load the object with its attributes and methods
        from caiman.source_extraction.cnmf import params as params
        from os.path import split
        from time import time
        
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
                if data_source.estimates.F_dff is None:
                    F = None
                else:
                    F = data_source.estimates.F_dff
                
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
           print('-------------------------------')
           print(f'Loaded data with {F.shape[0]} detected putative neurons')
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
            
            # Load the motion corrected movie fully into RAM
            print('Start loading the movie. This may take a moment...')
            ti = time()
            Yr = np.fromfile(movie_file, np.float32)
            movie = np.reshape(Yr,(image_dims[0], image_dims[1], recording_length)).transpose(1,0,2)
            print(f'Loaded in {time() - ti} seconds')
                
            #----Initialize the list of booleans for good neurons and a list with the indices    
            accepted_components = [None] * neuron_num # Create a list of neurons you want to keep or reject
            keep_neuron = [None] #Initialize the output of indices of the neurons to refit
            
            #----Sort the cells according to maximum instensity
            intensity_maxima = np.max(F,1) #Get maximum along the second dimension, e.g. within each row
            idx_max_int = np.argsort(-intensity_maxima) #The negative sign make the sorting go in descending order
            
            #Sort the data accordingly
            C = C[idx_max_int,:]
            S = S[idx_max_int,:]
            F = F[idx_max_int,:]
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
            movie_frame = movAx.imshow(movie[:,:,display_frame], cmap='gray', vmin=0, vmax=255) #The upper limit of pixel values
            #print(f'Plotted movie frame in {time() -st} seconds')
            neuron_mask = A[:,:,current_neuron] > 0 #Theshold to get binary mask
            neuron_contour = movAx.contour(neuron_mask, linewidths=0.5) #Overlay binary mask on movie frame
            
            #Then plot the de-noised cell activity alone
            pixel_intensity_scaling = 1/np.max(A[:,:,current_neuron])
            max_acti = np.max(C[current_neuron]) #Find the maximum of the denoised
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
            plt.setp(traceAx, ylim=(-10, round(np.max(F[current_neuron])+5))) #Scale y axis 
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
                movie_frame.set_data(movie[:,:,display_frame])
                
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
                #frame_slider.set_val(display_time)
                
                #Jump to the respective movie frame
                movie_frame.set_data(movie[:,:,display_frame])
                
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
                plt.setp(traceAx, ylim=(-10, round(np.max(F[current_neuron])+5))) #Scale y axis
                
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
        print(session_folder)
        output_folder = pjoin(session_folder[0], self.name) #Set output path directory to same location as data
        self.output_folder = self.name #Here this means the datatype folder only
        caiman_obj = load_CNMF(pjoin(output_folder, 'uncurated_caiman_results.hdf5'))
            
        #Run the GUI
        keep_neuron, frame_slider = neuron_selection_GUI(data_source = caiman_obj)
        #Only retain selected neurons
        caiman_obj.estimates.select_components(idx_components = keep_neuron)
        #Save the results
        caiman_obj.save(output_folder + '/caiman_results.hdf5')
    
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
