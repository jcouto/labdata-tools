#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
from os.path import join as pjoin
from io import StringIO
import numpy as np
import pandas as pd
import os
import sys
from glob import glob
from natsort import natsorted

LABDATA_FILE= pjoin(os.path.expanduser('~'),'labdatatools','preferences.json')

default_labdata_preferences = {'paths':[pjoin(os.path.expanduser('~'),'data')],
                               'path_format':'{subject}/{session}/{datatype}',
                               'remote_queue': None,
                               #'remote_queue': dict(remote='hoffman2.idre.ucla.edu',
                               #                     user='username'),
                               'rclone' : dict(drive = 'churchland_data',
                                               folder = 'data'),
                               'archives': None,
                               #'archives': [dict(drive = 'globus_shared',
                               #                         folder = 'data')],
                               'plugins_folder':pjoin(os.path.expanduser('~'),
                                                      'labdatatools','analysis')}

default_excludes = ['**.phy**',                # skip .phy folders
                    '**temp_wh.dat',           # skip kilosort temp_wh files
                    '**suite2p**data.bin',     # skip suite2p corrected files
                    '**.ipynb_checkpoints**',
                    '**._.DS_Store**',
                    '**.DS_Store**',
                    '**dummy**',
                    '**filtered_recording.ap.bin', # filtered recording from spks sorting
                    '**FakeSubject**']

def list_subjects():
    '''
    List subjects in the data path. Takes no arguments.
    Returns a pandas dataframe with the subject name and paths

Example:
    subjects = list_subjects()

    '''
    subjects = []
    for path in labdata_preferences['paths']:
        tmp = glob(pjoin(path,'*'))
        for t in tmp:
            if not os.path.isdir(t):
                continue
            if not os.path.basename(t) in [s['subject'] for s in subjects]:
                subjects.append(dict(subject=os.path.basename(t),paths = []))
            subjects[[s['subject'] for s in subjects].index(os.path.basename(t))]['paths'].append(t)
                
    return pd.DataFrame(subjects)

def list_files(subject = '', extension=''):
    '''
    Lists the files for a subject.
    Arguments:
        subject (string)
        extension (string)

    Example:
        files = list_files(extension = 'txt') # lists all text files
        files = list_files(subject = 'JC044', extension = 'txt') # lists all text files for JC044

    '''
    
    paths = []
    for server in labdata_preferences['paths']:
        if len(subject):
            if type(subject) is list:
                for s in subject:
                    paths.append(pjoin(server,'{subject}'.format(subject = s)))
            else:
                paths.append(pjoin(server,'{subject}'.format(subject = subject)))
        else:
            paths.append(server)

    files = []
    # recursive search
    for path in paths:
        tmp = [y for x in os.walk(path) 
                      for y in glob(pjoin(x[0], '*' + extension))]
        tmp = list(filter(os.path.isfile,tmp))
        for f in natsorted(np.array(tmp)):
            folder = f.replace(pjoin(path,''),'').split(os.path.sep)[0]
        
        for t in tmp:
            stats = os.stat(t)
            files.append(dict(filename = os.path.basename(t),
                              filepath = t,
                              relativepath = t.replace(path,''),
                              filesize = stats.st_size,
                              serverpath = '/'.join(
                                  os.path.normpath(t.replace(
                                      path.replace(subject,''),'')).split(
                                          os.path.sep)),
                              mtime = stats.st_mtime))
    return pd.DataFrame(files)


def get_filepath(subject,
                 session,
                 subfolders,
                 datapath = None,
                 filename = '*',
                 extension = '',
                 fetch = False,
                 **kwargs):
    '''
    Get a local filepath; retrieve from the remote if not present.
    Arguments:
       subject (string; required)
       session (string; required)
       subfolders (list of strings - use multiple levels if needed; required)
       datapath (the path of data; default is the first datapath in prefs)
       filename (the key in the filename; default '*')
       extension (default '')
       fetch (get data from the server if not present; default False)

    Example:
       session_files = get_filepath(subject = 'JC086',
                                    session = '20230113_131309',
                                    subfolders = ['DropletsTask'])

    '''
    if datapath is None:
        datapath = labdata_preferences['paths'][0]
    files = glob(pjoin(datapath,
                       subject,
                       session,
                       pjoin(*subfolders),
                       filename+extension))
    if len(files) == 1:
        files = files[0]
    if not len(files):
        files = None
    # If the file is not there return None but if fetch is TRUE go to the rclone to get it
    if fetch and files is None:
        print('Could not find file, trying to get it from the cloud')
        from .rclone import rclone_get_data
        rclone_get_data(subject=subject,
                        session = session,
                        datatype = subfolders[0],
                        includes = [filename+extension])
        # do the rclone stuff only once (if recursive it would just keep doing it)
        files = glob(pjoin(datapath,
                           subject,
                           session,
                           pjoin(*subfolders),
                           filename+extension))
        if len(files) == 1:
            files = files[0]
        if not len(files):
            files = None # return None if the file is not on the rclone nor in the local folder
    return files

def get_labdata_preferences(prefpath = None):
    ''' Reads the user parameters from the home directory.

    pref = get_labdata_preferences(filename)

    User parameters like folder location, file preferences, paths...

    '''
    if prefpath is None:
        prefpath = LABDATA_FILE
    preffolder = os.path.dirname(prefpath)
    if not os.path.exists(preffolder):
        os.makedirs(preffolder)
    if not os.path.isfile(prefpath):
        with open(prefpath, 'w') as outfile:
            json.dump(default_labdata_preferences, 
                      outfile, 
                      sort_keys = True, 
                      indent = 4)
            print('Saving default preferences to: ' + prefpath)
    with open(prefpath, 'r') as infile:
        pref = json.load(infile)
    for k in default_labdata_preferences:
        if not k in pref.keys():
            pref[k] = default_labdata_preferences[k]
    return pref

labdata_preferences = get_labdata_preferences()


