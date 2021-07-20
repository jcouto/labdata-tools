#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
from os.path import join as pjoin
from io import StringIO
import numpy as np
import pandas as pd
import os

LABDATA_FILE= pjoin(os.path.expanduser('~'),'.labdatatools')

default_preferences = {'paths':[pjoin(os.path.expanduser('~'),'data')],
                       'rclone' : dict(drive = 'churchland_data',
                                       folder = 'data')}

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
