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
from natsort import natsorted

LABDATA_FILE= pjoin(os.path.expanduser('~'),'labdatatools','preferences.json')

default_labdata_preferences = {'paths':[pjoin(os.path.expanduser('~'),'data')],
                               'path_format':'{subject}/{session}/{datatype}',
                               'slurm': dict(remote='hodgkin',
	                                     user='joao'),
                               'rclone' : dict(drive = 'churchland_data',
                                               folder = 'data'),
                               'plugins_folder':pjoin(os.path.expanduser('~'),
                                                      'labdatatools','analysis')}

def list_subjects():
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


def get_filepath(datapath,
                 subject,
                 session,
                 subfolders,
                 filename = '*',
                 extension = '',
                 fetch = False,
                 **kwargs):
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

