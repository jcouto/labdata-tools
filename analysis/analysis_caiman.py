###---The very basic intro---###
# This analysis plugin takes advantage of labdatatools to download raw data to
# be analyzed from the churchland gdrive and to upload the results of the
# analysis. Here's a use example in the terminal:
#
# labdata run caiman -a LO032 -s 20220215_114758 -d miniscope -- -decay_time 0.6
#
# This will download the miniscope folder from the specified subject and session
# and perform the caiman analysis with a decay_time of 0.6.

from labdatatools import BaseAnalysisPlugin
import argparse
from glob import glob
from os.path import join as pjoin
import os
import sys #To apppend the python path for the selection GUI



class AnalysisCaiman(BaseAnalysisPlugin):
    ##############
    fr = 30                          # movie frame rate
    decay_time = 0.6  #length of a typical transient in seconds
    pw_rigid = False # flag for pw-rigid motion correction
    gSig_filt = (3, 3) # size of filter, in general gSig (see below), change this one if algorithm does not work
    max_shifts = (5, 5) # maximum allowed rigid shift
    strides = (48, 48) # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24) # overlap between pathes (size of patch strides+overlap) maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3
    border_nan = 'copy' #Assignment of border pixels
    
    
    ##############
    p = 1               # order of the autoregressive system
    K = None            # upper bound on number of components per patch, in general None for 1p data
    gSig = (3, 3)       # gaussian width of a 2D gaussian kernel, which approximates a neuron
    gSiz = (13, 13)     # average diameter of a neuron, in general 4*gSig+1
    Ain = None          # possibility to seed with predetermined binary masks
    merge_thr = 0.7      # merging threshold, max correlation allowed
    rf = 80             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    stride_cnmf = 20    # amount of overlap between the patches in pixels
    #                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
    tsub = 2            # downsampling factor in time for initialization,
    #                     increase if you have memory problems
    ssub = 1            # downsampling factor in space for initialization,
    #                     increase if you have memory problems
    #                     you can pass them here as boolean vectors
    low_rank_background = None  # None leaves background of each patch intact,
    #                     True performs global low-rank approximation if gnb>0
    gnb = 0             # number of background components (rank) if positive,
    #                     else exact ring model with following settings
    #                         gnb= 0: Return background as b and W
    #                         gnb=-1: Return full rank background B
    #                         gnb<-1: Don't return background
    nb_patch = 0        # number of background components (rank) per patch if gnb>0,
    #                     else it is set automatically
    min_corr = 0.7       # min peak value from correlation image
    min_pnr = 8        # min peak to noise ration from PNR image
    ssub_B = 2          # additional downsampling factor in space for background
    ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor
         
    
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

    def parse_arguments(self, arguments = []): #This is currently not implemented for local executions!
        '''Create parser object to be able to easily execute the analysis in the command line later.'''
        parser = argparse.ArgumentParser(
                description = '''
                    Identification of individual neurons in calium imaging data.
                    Actions are: run, curate
                    ''',
                usage = 'caiman -a <subject> -s <session> -- run|curate <PARAMETERS>') #To be adapted later

        #parser.add_argument('action',
        #        action = 'store', type = str, help = "action to perform (RUN the analysis on a dataset, CURATE the extracted neurons)")
                                #The 0th argument should always be the action that needs to be performed. This will also help strcturing the call
                                #to the function and display the help hierarchically.
        parser.add_argument('--round-num', #There is a convention to use a dash rather than an underscore.
                action = 'store', default = 'second', type = str, help = "Specify the caiman extraction round.")

        args = parser.parse_args(arguments[1:])

        self.round_num = args.round_num
        #Argparse converts dashes to underscores in optional arguments (not in the positional one).

    def _run(self):
        
        import caiman as cm
        from caiman.source_extraction import cnmf
        from caiman.utils.utils import download_demo
        from caiman.utils.visualization import inspect_correlation_pnr
        from caiman.motion_correction import MotionCorrect
        from caiman.source_extraction.cnmf import params as params
        
        import os #For all the file path manipulations
        from time import time # For time logging
        import sys #To apppend the python path for the selection GUI
        import numpy as np
        
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                 n_processes=2,  # number of process to use, if you go out of memory try to reduce this one, changed to 2-cores temporarily cause my pc is hot garbo
                                                 single_thread=False)


        #File Selection:
        session_folders = self.get_sessions_folders() #Go thorugh the folders
        print(session_folders)
        fnames= sorted(glob(os.path.join(session_folders[0], self.datatypes[0], 'binned_*.avi')))
        #print(fnames)
        
        # Initial CNMF params
        opts = params.CNMFParams(params_dict={
            'fnames': fnames,
            'fr': self.fr,
            'decay_time': self.decay_time,
            'pw_rigid': self.pw_rigid,
            'max_shifts': self.max_shifts,
            'gSig_filt': self.gSig_filt,
            'strides': self.strides,
            'overlaps': self.overlaps,
            'max_deviation_rigid': self.max_deviation_rigid,
            'border_nan': self.border_nan
        })
        
        mc_start_time = time()
    
        # Motion Correction Step:
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)
        
        fname_mc = mc.fname_tot_els if self.pw_rigid else mc.fname_tot_rig
        
        if self.pw_rigid:
            bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                         np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
        else:
            bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)

    
        bord_px = 0 if self.border_nan == 'copy' else bord_px
        fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                                   border_to_0=bord_px)
        
        mc_end_time = time()
        #Display elapsed time
        print(f"Motion Correction Finished in: {round(mc_end_time - mc_start_time)} s.")
        
        #Save the shifts  
        rigid_shifts = np.array(mc.shifts_rig) # Retrieve shifts from mc object
        
        outputPath = os.path.join(session_folders[0], self.name) #Set path directory to same location as data
        
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
        
        np.save(outputPath + '/rigid_shifts', rigid_shifts) #Save the np array to npy file
         

        #Load Memory Mappable File
        Yr, dims, T = cm.load_memmap(fname_new, mode='r+')
        images = Yr.T.reshape((T,) + dims, order='F')
        
        
        
        opts.change_params(params_dict={'dims': dims,
                                'method_init': 'corr_pnr',  # use this for 1 photon
                                'p': self.p,
                                'K': self.K,
                                'gSig': self.gSig,
                                'gSiz': self.gSiz,
                                'merge_thr': self.merge_thr,
                                'rf': self.rf,
                                'stride': self.stride_cnmf,
                                'tsub': self.tsub,
                                'ssub': self.ssub,
                                'low_rank_background': self.low_rank_background,
                                'nb': self.gnb,
                                'nb_patch': self.nb_patch,
                                'min_corr': self.min_corr,
                                'min_pnr': self.min_pnr,
                                'ssub_B': self.ssub_B,
                                'ring_size_factor': self.ring_size_factor,
                                'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                'only_init': True,    # set it to True to run CNMF-E
                                'update_background_components': True,  # sometimes setting to False improve the results
                                'normalize_init': False,               # just leave as is
                                'center_psf': True,                    # leave as is for 1 photon
                                'del_duplicates': True,                # whether to remove duplicates from initialization
                                'border_pix': bord_px})                # number of pixels to not consider in the borders)

        
        
        #Remove pixels with basically zero intensity but very few 

        medProj = np.median(images, axis=0, keepdims=True)
        median_bool = np.squeeze(medProj < 1)
        for k in range(images.shape[0]):
            temp = images[k,:,:]
            temp[median_bool] = 0.0001
            images[k,:,:] = temp
            
            
        #Compute spatiotemporal correlations on images:    
            
        corr_image_start = time()
        cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=self.gSig[0], swap_dim=False)
        #Compute the correlation and pnr image on every frame. This takes longer but will yield
        #The actual correlation image that can be used later to align other sessions to this session
        corr_image_end = time()
        
        
        print(f"Computed Correlations in: {corr_image_end - corr_image_start} s.")
        
        np.save(outputPath + '/spatio_temporal_correlation_image', cn_filter)
        np.save(outputPath + '/median_projection', medProj)
        # if your images file is too long this computation will take unnecessarily
        # long time and consume a lot of memory. Consider changing images[::1] to
        # images[::5] or something similar to compute on a subset of the data
        
        # inspect the summary images and set the parameters
        inspect_correlation_pnr(cn_filter, pnr)
        
        # print parameters set above, modify them if necessary based on summary images
        print(self.min_corr) # min correlation of peak (from correlation image)
        print(self.min_pnr)  # min peak to noise ratio
        
        # Shuts down parallel pool and restarts
        dview.terminate()
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                         n_processes=2,  # i have hot garbo computer
                                                         single_thread=False)
        
       # Run CNMF-E on Patches:
        cnmfe_start_time = time()
        
        cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=self.Ain, params=opts)
        cnm.fit(images)
        
        #Display elapsed time
        cnmfe_end_time = time()
        print(f"Ran Initialization and Fit CNMFE Model in: {round(cnmfe_end_time - cnmfe_start_time)} s.")
        
        # Save first round of results
        cnm.save(outputPath + '/firstRound.hdf5')  

 
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
