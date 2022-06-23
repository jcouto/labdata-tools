#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from subprocess import check_output,Popen,PIPE,STDOUT
from .utils import *
from tqdm import tqdm

def rclone_list_subjects():
    '''
    Lists the subjects on the google drive
    Returns a pandas dataframe with the kist of subjects if the subjects are 
the first level of the folder hierarchy.
    '''
    out = check_output('rclone lsd {drive}:{folder}'.format(**labdata_preferences['rclone']).split(' ')).decode("utf-8")
    out = out.splitlines()
    subjects = [o.split(' ')[-1] for o in out]
    if len(subjects):
        return subjects
    else:
        return None
    
def rclone_list_sessions(subject):
    out = check_output(
        'rclone lsd {drive}:{folder}/{subject}'.format(
            **labdata_preferences['rclone'],
            subject = subject).split(' ')).decode("utf-8")
    out = out.splitlines()
    sessions = [o.split(' ')[-1] for o in out]
    if len(sessions):
        return sessions
    else:
        return None

def rclone_list_files(subject = '', filters = [],
                      includes = [],
                      excludes = []):
    '''
    Gets a list of all files in the remote.
    Specify a subject to get only the first level.
    '''
    cmd = 'rclone ls {drive}:{folder}/{subject}'.format(
        subject=subject,
        **labdata_preferences['rclone'])
    if len(includes):
        for i in includes:
            cmd += ' --include "{0}"'.format(i)
    if len(excludes):
        for i in excludes:
            cmd += ' --exclude "{0}"'.format(i)

    #print(cmd,flush=True)
    out = check_output(cmd.split(' ')).decode("utf-8")
    files = []
    for a in out.split('\n'):
        a = a.strip(' ').split(' ')
        if len(a)>1:
            a[1] = ' '.join(a[1:])
            sz = int(a[0])
            fname = a[1]
            dirname = os.path.dirname(fname)
            tmp = dirname.split('/')
            sub = None
            session = None
            datatype = None
            if len(tmp)>=1:
                s = 0
                if subject == '':
                    s = 1
                if len(tmp)>= s+1:
                    if subject == '':
                        sub = tmp[0+s-1]
                    else:
                        sub = subject # main folder is the subject
                    session = tmp[0+s]
                    if len(tmp)>=s+2:
                        datatype = tmp[1+s]
            include = False
            for inn in filters:
                if inn in fname:
                    include = True
            if not len(filters):
                include = True
            if datatype is None: # for files on the session level
                 datatype = '.'
            if include:
                files.append(dict(filename=os.path.basename(fname),
                                  filesize = int(sz),
                                  filepath = fname,
                                  dirname = dirname,
                                  subject = sub,
                                  session = session,
                                  datatype = datatype))
    return pd.DataFrame(files)

def rclone_get_data(path_format = None,
                    includes = [],
                    excludes = [],
                    ipath = 0,
                    verbose = True,
                    **kwargs):
    '''
    Fetch data from the data server.
    
Note:

    TODO: Add examples for how to filter. 
    '''
    if path_format is None:
        path_format = labdata_preferences['path_format']

    fmts = path_format.split('/')
    keys = dict(labdata_preferences['rclone'],
                path = labdata_preferences['paths'][ipath],
                **kwargs)
    for fmt in fmts:
        fmt = fmt.strip('{').strip('}')
        if not fmt in keys.keys():
            keys[fmt] = '*'
    for fmt in fmts[::-1]:
        fmt = fmt.strip('{').strip('}')
        if keys[fmt] == '*':
            fmts.pop()
        else:
            break
    
    local_path = pjoin('{path}',*fmts)  # build local path, so it is OS independent
    keys['drive_path'] = '/'.join(fmts).format(**keys)
    keys['local_path'] = local_path.format(**keys)
    cmd = 'rclone copy --progress {drive}:{folder}/{drive_path} {local_path}'.format(**keys)
    if len(includes):
        for i in includes:
            cmd += ' --include {0}'.format(i)
    if len(excludes):
        for i in excludes:
            cmd += ' --exclude {0}'.format(i)
    if verbose:
        print(cmd)
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

def rclone_upload_data(subject='',
                       session = None,
                       datatype = None,
                       path_idx = 0,
                       bwlimit = None,
                       excludes = ['.phy'],
                       overwrite = False):
    # this needs a pipe
    localpath = labdata_preferences['paths'][path_idx]
    if len(subject):
        localpath = pjoin(localpath,subject)
        subject = '/' + subject
    remotepath = subject
    
    if not session is None:
        localpath = pjoin(localpath,session)
        remotepath += '/'+session
        if not datatype is None:  # needs a session
            localpath = pjoin(localpath,datatype)
            remotepath += '/'+datatype
        
    command = 'rclone copy --progress {path} {drive}:{folder}{remote}'.format(
        remote=remotepath,
        path = localpath,
        **labdata_preferences['rclone'])
    if not bwlimit is None:
        command += ' --bwlimit {0}M'.format(bwlimit)
    if not overwrite:
        command += ' --ignore-existing'
    if len(excludes):
        for i in excludes:
            command += ' --exclude {0}'.format(i)
    print(command)
    process = Popen(command, shell=True, 
                    stdout=PIPE, stderr=STDOUT,
                    universal_newlines = False)
    while True:
        nextline = process.stdout.readline()
        nextline = nextline.decode()
        if nextline == '' and process.poll() is not None:
            break
        #if 'ETA' in nextline:
        print('\r'+nextline,
              end='',
              flush=True) # decode does not play nice with "\r"

    output = process.communicate()[0]
    exitCode = process.returncode
