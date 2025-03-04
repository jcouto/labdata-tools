#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from subprocess import check_output,Popen,PIPE,STDOUT
from .utils import *
from tqdm import tqdm

def get_archived_list(pref = labdata_preferences):
    preffolder = os.path.dirname(LABDATA_FILE)
    archived_files = []
    if 'archives' in pref.keys():
        if not pref['archives'] is None:
            for iarchive in range(len(pref['archives'])):
                filename = pjoin(preffolder,'archive_list_{drive}_{folder}.csv'.format(**pref['archives'][iarchive]))
                pref['archives'][iarchive]['file_list'] = filename
                if not os.path.exists(filename):
                    print('No list for archive [{0}]'.format(pref['archives'][iarchive]['drive']),flush = True)
                    try:
                        
                        res = check_output('rclone copyto --progress {drive}:/archive_list.csv {filename}'.format(
                            drive=labdata_preferences['archives'][iarchive]['drive'],
                            filename = filename),
                                           shell=True)
                    except:
                        print('Fetching list of all files.')
                        print('... this will take a while...',flush=True)
                        from .rclone import rclone_list_files
                        files = rclone_list_files(remote = pref['archives'][iarchive])
                        print('Saving to: {0}'.format(filename),flush=True)
                        files.to_csv(filename,index = False)
                files = pd.read_csv(filename)
                files['remote_drive'] = labdata_preferences['archives'][iarchive]['drive']
                files['remote_folder'] = labdata_preferences['archives'][iarchive]['folder']
                archived_files.append(files)
    if len(archived_files):
        return pd.concat(archived_files)
    else: return None
    
def rclone_list_subjects(remote = None):
    '''
    Lists the subjects on the google drive
    Returns a pandas dataframe with the list of subjects if the subjects are 
the first level of the folder hierarchy.
    '''
    subjects = []
    if remote is None:
        remote = labdata_preferences['rclone']
        if 'archives' in labdata_preferences.keys():
            archivefiles = get_archived_list()
            if not archivefiles is None:
                subjects = list(filter(lambda x:type(x) is str,
                                       archivefiles.subject.drop_duplicates().values))
    out = check_output('rclone lsd {drive}:{folder}'.format(**remote).split(' ')).decode("utf-8")
    out = out.splitlines()
    subjects += [o.split(' ')[-1] for o in out]
    if len(subjects):
        return np.unique(subjects)
    else:
        return None
    
def rclone_list_sessions(subject,remote = None):
    sessions = []
    if remote is None:
        remote = labdata_preferences['rclone']
        if 'archives' in labdata_preferences.keys():
            archivefiles = get_archived_list()
            if not archivefiles is None:
                sessions  = [i for i in archivefiles[
                    archivefiles.subject == subject].session.drop_duplicates().values]
    try:
        out = check_output(
            'rclone lsd {drive}:{folder}/{subject}'.format(
                **remote,
                subject = subject).split(' ')).decode("utf-8")
        out = out.splitlines()
        sessions += [o.split(' ')[-1] for o in out]
    except:
        pass
    if len(sessions):
        return np.unique(sessions)
    else:
        return None

def rclone_list_files(subject = '', filters = [],
                      includes = [],
                      excludes = [],
                      remote = None):
    '''
    Gets a list of all files in the remote.
    Specify a subject to get only the first level.
    '''
    archivefiles = []
    if remote is None:
        remote  = labdata_preferences['rclone']
        if 'archives' in labdata_preferences.keys():
            archivefiles = get_archived_list()
            if len(subject):
                archivefiles = archivefiles[archivefiles.subject == subject]
                
    cmd = 'rclone ls {drive}:{folder}/{subject}'.format(
        subject=subject,
        **remote)
    if len(includes):
        for i in includes:
            cmd += ' --include "{0}"'.format(i)
    if len(excludes):
        for i in excludes:
            cmd += ' --exclude "{0}"'.format(i)

    #print(cmd,flush=True)
    try:
        out = check_output(cmd.split(' ')).decode("utf-8") #FIXME: this returns a nonzero exit status and a CalledProcessError if there are no directories found. Problematic when repeating over archives.
    except:
        out = ''
    files = []
    for a in out.split('\n'):
        a = a.strip(' ').split(' ')
        if len(a)>1:
            a[1] = ' '.join(a[1:])
            sz = int(a[0])
            fname = a[1]
            if not subject  == '':
                fname = subject+'/'+fname
            dirname = os.path.dirname(fname)
            tmp = dirname.split('/')
            sub = None
            session = None
            datatype = None
            if len(tmp)>=1:
                s = 1
                if len(tmp)>= s+1:
                    if subject == '':
                        sub = tmp[0+s-1]
                    else:
                        sub = subject # main folder is the subject
                    session = tmp[0+s]
                    if len(tmp)>=s+2:
                        datatype = tmp[1+s]
            if datatype is None: # for files on the session level
                 datatype = '.'
            files.append(dict(filename=os.path.basename(fname),
                              filesize = int(sz),
                              filepath = fname,
                              dirname = dirname,
                              subject = sub,
                              session = session,
                              datatype = datatype,
                              remote_drive = remote['drive'],
                              remote_folder = remote['folder']))
            
    if len(archivefiles):
        files = pd.concat([archivefiles,pd.DataFrame(files)])
    else:
        files = pd.DataFrame(files)
    if len(files) and len(filters):
        idx = []
        for filt in filters:

            for i,fname in enumerate(files.filepath.values):
                if filt in fname:
                    idx.append(i)
        idx = np.unique(idx)
        files = files.iloc[idx]
            
    return files

def rclone_get_data(path_format = None,
                    includes = [],
                    excludes = [],
                    ipath = 0,
                    overwrite = False,
                    verbose = True,
                    remote = None,
                    **kwargs):
    '''
    Fetch data from the data server.
    
Note:

    TODO: Add examples for how to filter. 
    '''
    if path_format is None:
        path_format = labdata_preferences['path_format']

    fmts = path_format.split('/')
    if remote is None:
        #print("Get data called but no remote provided. Going to get recursively from all remotes possible.")
        if 'archives' in labdata_preferences.keys():
            remotes = [labdata_preferences['rclone']] + labdata_preferences['archives']
        else:
            remotes = [labdata_preferences['rclone']]
        for r in remotes:
            rclone_get_data(path_format = path_format,
                            includes = includes,
                            excludes = excludes,
                            ipath = ipath,
                            overwrite = overwrite,
                            verbose = verbose,
                            remote = r,
                            **kwargs)
        return
    keys = dict(remote,
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
    if '*' in keys['local_path']:
        # then add it to the includes instead
        toinclude = os.path.basename(keys['local_path'])
        if not '*' in toinclude:
            raise(ValueError("Can only add the wildkey to the last folder name" + keys['local_path']))
        keys['local_path'] = os.path.dirname(os.path.abspath(keys['local_path']))
        keys['drive_path'] = os.path.dirname(keys['drive_path'])
        includes.append(toinclude)
    cmd = 'rclone copy --progress {drive}:{folder}/{drive_path} {local_path}'.format(
        **keys)
    if len(includes):
        for i in includes:
            cmd += ' --include "{0}"'.format(i)
    if len(excludes):
        for i in excludes:
            cmd += ' --exclude "{0}"'.format(i)
    if not overwrite:
        cmd += ' --ignore-existing'
    if verbose:
        print(cmd)
    process = Popen(cmd, shell=True, 
                    stdout=PIPE, stderr=PIPE,
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
                       max_transfer = None,
                       excludes = default_excludes,
                       includes = [],
                       overwrite = False,
                       add_pacer_options = True,
                       check_archives = True,
                       remote = None):
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

    # check that the files are not already on an archive first.
        
    if remote is None:
        remote = labdata_preferences['rclone']

    command = 'rclone copy --progress {path} {drive}:{folder}{remote}'.format(
        remote=remotepath,
        path = localpath,
        **remote)
    if add_pacer_options:
        command += ' --drive-pacer-min-sleep 10ms --drive-pacer-burst 1000'
    if not bwlimit is None:
        command += ' --bwlimit {0}M'.format(bwlimit)
    if not max_transfer is None:
        command += ' --max-transfer={0}G --cutoff-mode=cautious'.format(max_transfer)
    if not overwrite:
        command += ' --ignore-existing'
    if len(excludes):
        for i in excludes:
            command += ' --exclude "{0}"'.format(i)
    if len(includes):
        for i in includes:
            command += ' --include "{0}"'.format(i)

    if check_archives:
        if 'archives' in labdata_preferences.keys():
            for a in labdata_preferences['archives']:
                command += ' --compare-dest {0}:{1} --size-only'.format(a['drive'],a['folder'])
    print(command)
    process = Popen(command, shell=True, 
                    stdout=PIPE, stderr=PIPE,
                    universal_newlines = False)
    while True:
        nextline = process.stdout.readline()
        nextline = nextline.decode()
        if nextline == '' and process.poll() is not None:
            break
        #if 'ETA' in nextline:
        print(nextline,end = '\r') # decode does not play nice with "\r"
        
    output = process.communicate()[0]
    exitCode = process.returncode
    return exitCode
