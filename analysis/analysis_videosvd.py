#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: LY 2024

Compute SVD of videos via labdata tools: currently can compute SVD of original
full video, and can create a motion energy video and then compute the SVD.

Sample usage: labdata run videosvd -a LY018 -s 20231109_141923 -d chipmunk -- compute_video_svd

    
"""
from labdatatools.utils import *
from labdatatools import BaseAnalysisPlugin
import argparse
import json
import sys
import numpy as np
import cv2
import os 
import time
from glob import glob
from tqdm import tqdm
   

class AnalysisVideosvd(BaseAnalysisPlugin):

    def __init__(self,subject,
                 session = None,
                 datatypes = [''],
                 includes = [''],
                 excludes = [''],
                 bwlimit = None,
                 overwrite = False,
                 **kwargs):

        #Inherit methods and attributes from the BaseAnalysisPlugin
        super(AnalysisVideosvd,self).__init__(
            subject = subject,
            session = session,
            datatypes = datatypes,
            includes = includes,
            excludes = excludes,
            bwlimit = bwlimit,
            overwrite = overwrite,
            **kwargs)
        self.name = 'videosvd'
        self.params =  None #These will be the parameters passed via the arguments 
        self.output_folder = "analysis" #Here this means the datatype folder only

    def parse_arguments(self, arguments = []): #This is currently not implemented for local executions!
        '''Create parser object to be able to easily execute the analysis in the command line later.'''
        parser = argparse.ArgumentParser(
                description = '''
                    Process .avi file and generate motion energy video. 
                    Actions are: compute_video_svd, compute_motion_energy_svd
                    ''',
                usage = 'videosvd -a <subject> -s <session> -d <datatype> -- generate_video <PARAMETERS>')

        parser.add_argument('action',
                            action='store', type=str, help = "Input action to perform, see above.")

        args = parser.parse_args(arguments[1:])

        self.action = args.action
        print(self.action)

        if self.action == 'compute_video_svd': #Method to generate motion energy video
            self._run = self._run_compute_video_svd
        elif self.action == "compute_motion_energy_svd":
            self._run = self._run_compute_motion_energy_svd
        else:
            raise(ValueError('Available command are: compute_video_svd, compute_motion_energy_svd'))
#---------------------------------------------------------------------------
            
    def _run_compute_video_svd(self):
        """
        Computes the video SVD for original video.
        """
       
        from wfield import load_stack, approximate_svd, chunk_indices
        import numpy as np
        import os
        from glob import glob
        from tqdm import tqdm


    

        session_folder = self.get_sessions_folders() #Go thorugh the folders
        fnames = glob(os.path.join(session_folder[0], self.datatypes[0], '*.avi'))[0] #These are the complete video paths
        output_folder = pjoin(session_folder[0], self.output_folder) #Set path directory to analysis folder
        output_file = os.path.join(os.path.split(os.path.split(fnames)[0])[0], self.output_folder, os.path.splitext(os.path.split(fnames)[1])[0][:-9] + '_video_SVD.npy')
    

        if not os.path.exists(output_folder):
           os.mkdir(output_folder)
    
    
       
       #Start by obtaining the svd from the full video
        print('Starting video svd...')
       
        dat = load_stack(fnames)
        chunkidx = chunk_indices(len(dat),chunksize=256)
        frame_averages = []
       
        for on,off in tqdm(chunkidx, desc='Computing average.'):
             frame_averages.append(dat[on:off].mean(axis = 0))
        frames_average = np.stack(frame_averages).mean(axis = 0)
        U,SVT = approximate_svd(dat, frames_average,nframes_per_bin=2)

        np.save(output_file, [U, SVT]) #saves video_SVD file in output folder
    
    
    def _run_compute_motion_energy_svd(self):
        
        from wfield import load_stack, approximate_svd, chunk_indices
        import numpy as np
        import os
        from glob import glob
        from tqdm import tqdm
        
        session_folder = self.get_sessions_folders() #Go thorugh the folders
        fnames = glob(os.path.join(session_folder[0], self.datatypes[0], '*.avi'))[0] #These are the complete video paths
        output_folder = pjoin(session_folder[0], self.output_folder) #Set path directory to analysis folder
        output_file = os.path.join(os.path.split(os.path.split(fnames)[0])[0], self.output_folder, os.path.splitext(os.path.split(fnames)[1])[0][:-9] + '_motion_energy_SVD.npy')

        print(fnames)
        print(output_file)
        
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        
        
        '''Generates a motion energy video from the provided input.
        assumes input video is uint8! Use the ugly while(True) implementation
        here because some of the video headers might be inaccurate
        This is rather inefficient for now...'''
        
        
        cap = cv2.VideoCapture(fnames)
        frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #Number of frames 
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_dims = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        
        me_video_file = os.path.join(os.path.split(os.path.split(fnames)[0])[0], self.output_folder, os.path.splitext(os.path.split(fnames)[1])[0][:-9] + '_motion_energy.avi')

        wrt = cv2.VideoWriter(me_video_file, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, frame_dims)
        previous_frame = np.array([])
        
        timer = time.time()
        sporadic_report = 0
        # k = 0
        for k in range(frame_number):
        # while(True):
            success, f = cap.read()
            
            if not success:
                print(f'Something_happend at frame {k}. Check the movie!')
            
            frame = np.array(f[:,:,1], dtype=float) #The video is gray-scale, convert to float to be able to record negative values
            
            if previous_frame.shape[0] > 0:
                motion_energy = np.abs(frame -previous_frame).astype(np.uint8)
            else:
                motion_energy = np.zeros([frame_dims[1], frame_dims[0]]).astype(np.uint8)
            
            wrt.write(np.stack((motion_energy, motion_energy, motion_energy), axis=2))
            previous_frame = frame
            
            sporadic_report = sporadic_report + 1
            if sporadic_report == 5000:
                print(f'Wrote {k} frames to file.')
                sporadic_report = 0
           # k = k+1   
            
        wrt.release()
        print(f'Completed in {time.time() - timer} seconds.')
        print('--------------------------------------------')
        
           #Now do the svd on motion energy video
           
        print('Starting the svd for the motion energy video')

        dat = load_stack(me_video_file)
        frames_average = np.zeros([dat.shape[1], dat.shape[2], dat.shape[3]])
        U,SVT = approximate_svd(dat, frames_average,nframes_per_bin=2)
        np.save(output_file, [U, SVT])
        
        os.remove(me_video_file) #deletes the ME video after computing the SVD
