#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
from os.path import join as pjoin
from io import StringIO
import numpy as np
import pandas as pd
import os
from glob import glob

LABDATA_FILE= pjoin(os.path.expanduser('~'),'.labdatatools')

default_preferences = {'paths':[pjoin(os.path.expanduser('~'),'data')],
                       'rclone' : dict(drive = 'churchland_data',
                                       folder = 'data')}

def get_filepath(datapath,
                 subject,
                 session,
                 subfolders,
                 filename = '*',
                 extension = '',
                 fetch = False):
    '''Get a local filepath by extension'''
    files = glob(pjoin(datapath,
                       subject,
                       session,
                       pjoin(*subfolders),
                       filename+extension))
    if len(files) == 1:
        files = files[0]
    if not len(files):
        files = None
    if fetch and files is None:
        print('Could not find file, trying to get it from the cloud')
        from .rclone import rclone_get_data
        rclone_get_data(subject=subject,
                        session = session,
                        datatype = subfolders[0],
                        includes = [filename+extension])
        files = get_filepath(datapath,
                             subject,
                             session,
                             subfolders,
                             filename,
                             extension)
    return files

def get_preferences(prefpath = None):
    ''' Reads the user parameters from the home directory.

    pref = get_preferences(filename)

    User parameters like folder location, file preferences, paths...

    '''
    if prefpath is None:
        prefpath = LABDATA_FILE
    preffolder = os.path.dirname(prefpath)
    if not os.path.isfile(prefpath):
        with open(prefpath, 'w') as outfile:
            json.dump(default_preferences, 
                      outfile, 
                      sort_keys = True, 
                      indent = 4)
            print('Saving default preferences to: ' + prefpath)
    with open(prefpath, 'r') as infile:
        pref = json.load(infile)
    return pref

preferences = get_preferences()

