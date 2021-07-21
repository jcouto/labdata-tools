#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from subprocess import check_output,Popen,PIPE,STDOUT
from .utils import *
from tqdm import tqdm

def rclone_list_subjects():
    '''
    Lists the subjects on the google drive
    Returns a pandas dataframe with the kist of subjects
    '''
    out = check_output('rclone lsd {drive}:{folder}'.format(**preferences['rclone']).split(' ')).decode("utf-8")
    out = out.splitlines()
    subjects = [o.split(' ')[-1] for o in out]
    if len(subjects):
        return subjects
    else:
        return None
    
def rclone_list_sessions(subject):
    out = check_output(
        'rclone lsd {drive}:{folder}/{subject}'.format(
            **preferences['rclone'],
            subject = subject).split(' ')).decode("utf-8")
    out = out.splitlines()
    sessions = [o.split(' ')[-1] for o in out]
    if len(sessions):
        return sessions
    else:
        return None

def rclone_list_files(subject = '', filters = []):

    cmd = 'rclone ls {drive}:{folder}/{subject}'.format(
        subject=subject,
        **preferences['rclone'])
    #if len(includes):
    #    for i in includes:
    #        cmd += ' --include {0}'.format(i)
    #if len(excludes):
    #    for i in excludes:
    #        cmd += ' --exclude {0}'.format(i)

    out = check_output(cmd.split(' ')).decode("utf-8")
    print(cmd)
    files = []
    for a in out.split('\n'):
        a = a.strip(' ')
        if len(a):
            sz = re.search(r'\d+', a).group()
            fname = a.replace(sz,'').strip(' ')
            dirname = os.path.dirname(fname)
            tmp = dirname.split('/')
            session = None
            datatype = None
            if len(tmp)>=2:
                s = 0
                if subject == '':
                    s = 1
                session = tmp[0+s]
                datatype = tmp[1+s]
            include = False
            
            for inn in filters:
                if inn in fname:
                    include = True
            if not len(filters):
                include = True
            if include:
                files.append(dict(filename=os.path.basename(fname),
                                  filesize = int(sz),filepath = fname,
                                  dirname = dirname,
                                  session = session,
                                  datatype = datatype))
    return pd.DataFrame(files)

def rclone_upload_data(subject=''):
    # this needs a pipe
    if not len(subject):
        subject = '/' + subject
    command = 'rclone copy --progress {path} {drive}:{folder}{subject}'.format(
        subject=subject,
        path = preferences['paths']['serverpaths'][0],
        **preferences['rclone'])
    process = Popen(command, shell=True, 
                    stdout=PIPE, stderr=STDOUT,
                    universal_newlines = False)
    while True:
        nextline = process.stdout.readline()
        nextline = nextline.decode()
        if nextline == '' and process.poll() is not None:
            break
        print(nextline, end='',flush=True) # decode does not play nice with "\r"
    output = process.communicate()[0]
    exitCode = process.returncode

def rclone_get_data(subject='', session = '', datatype = '', includes = [], excludes = []):
    '''
    Fetch data from the data server.
    
Note:

    TODO: Add examples for how to filter. 
    '''
    if len(subject):
        subject = '/' + subject
        if len(session):
            session = '/' + session
        else:
            if len(datatype):
                print('DATATYPE does not work without SESSION. Use filters.')
                datatype = ''
        if len(datatype):
            datatype = '/' + datatype
    else:
        if len(session):
            print('SESSION does not work without SUBJECT. Use filters.')
            session = ''
        if len(datatype):
            print('DATATYPE does not work without SUBJECT. Use filters.')
            datatype = ''
            
    cmd = 'rclone copy --progress {drive}:{folder}{subject}{session}{datatype} {path}{subject}{session}{datatype}'.format(
        subject=subject,
        path = preferences['paths'][0],
        session = session,
        datatype = datatype,
        **preferences['rclone'])
    if len(includes):
        for i in includes:
            cmd += ' --include {0}'.format(i)
    if len(excludes):
        for i in excludes:
            cmd += ' --exclude {0}'.format(i)

    process = Popen(cmd, shell=True, 
                    stdout=PIPE, stderr=STDOUT,
                    universal_newlines = False)
    with tqdm(position=0, leave = True) as pbar:
        value = 0
        total = 1
        while True:
            nextline = process.stdout.readline()
            nextline = nextline.decode()
            #print(nextline, end='',flush=True) # decode does not play nice with "\r"
            if 'Transferred:' in nextline and '%' in nextline:
                tmp = [i.strip(',') for i in nextline.split()]
                progress = [int(i) for i in tmp if i.isdigit()]
                if len(progress) >= 2:
                    if not total == progress[1]:
                        total = progress[1]
                        pbar.total = total
                    if not value == progress[0]:
                        pbar.update(progress[0]-value)
                        value = progress[0]
            if nextline == '' and process.poll() is not None:
                break
    output = process.communicate()[0]
    return process.returncode
