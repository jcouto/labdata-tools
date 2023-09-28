from .utils import *
from .rclone import rclone_list_files

def clean_local_files(subject = None,
                      checksum = True,
                      dry_run = False,
                      keep_recent_weeks = 5,
                      exceptions = ['Session Settings',
                                    'Session Data'] + default_excludes):
    from tqdm import tqdm  # make this optional

    to_delete = []
    to_keep = []
    subjects = list_subjects()
    if not subject is None:
        if '*' in subject:
            # then get all subjects that qualify
            k = subject.replace('*','')
            subjects = [s for i,s in subjects.iterrows() if k in s.subject]
            if len(subjects):
                subjects = pd.DataFrame(subjects)
            else:
                print("No local subject fit the key: {0}".format(subject))
        else:
            subjects = subjects[subjects.subject == subject]

    for i,s in subjects.iterrows():
        print('Cleaning data for {0}'.format(s.subject))
        localfiles = list_files(s.subject)
        if not len(localfiles):
            continue
        # don't look at the files if in exceptions
        exceptionidx = []
        for i,f in enumerate(localfiles.filepath):
            for e in exceptions:
                if e in f:
                    exceptionidx.append(i)
        localfiles = localfiles.drop(localfiles.index[exceptionidx])
        if localfiles.shape[0]<1:
            print('No data for subject {0}'.format(s.subject))
            continue
        localfiles['localfolder'] = localfiles.filepath.map(os.path.dirname)
        localfiles['serverfolder'] = localfiles.serverpath.map(os.path.dirname)
        if checksum:
            localfolders = localfiles[['localfolder','serverfolder']].drop_duplicates()
            # check folder by folder with rclone
            for j,f in tqdm(localfolders.iterrows()):
                from subprocess import getstatusoutput
                cmd = 'rclone check --one-way '+ f.localfolder + ' {drive}:{folder}/{fpath}'.format(**labdata_preferences['rclone'], 
                                                                                                    fpath = f.serverfolder)
                code,out = getstatusoutput(cmd)
                if ': 0 differences found' in out and not 'ERROR' in out:
                    # delete files
                    to_delete.append(localfiles[localfiles.localfolder == f.localfolder])
                else:
                    to_keep.append(localfiles[localfiles.localfolder == f.localfolder])
                    print(out)
        else:
            try:
                remotefiles = rclone_list_files(s.subject)
            except:
                print('Subject [{0}] not on the remote.'.format(s.subject),
                      flush=True)
                continue
            for j,l in localfiles.iterrows():
                r = remotefiles[remotefiles.filepath == l.serverpath]
                if r.shape[0]>0:
                    r = r.iloc[0]
                if r.shape[0]>0 and r.filesize == l.filesize:
                    to_delete.append(l)
                else:
                    # file not in remote
                    to_keep.append(l)
                    if r.shape[0]>0:
                        print('File {0}, local:{1} remote:{2} bytes'.format(l.filepath,l.filesize,r.filesize))
    if checksum:
        if not len(to_keep):
            to_keep = pd.DataFrame(to_keep)
        else:
            to_keep = pd.concat(to_keep)
        if not len(to_delete):
            to_delete = pd.DataFrame(to_delete)
        else:
            to_delete = pd.concat(to_delete)
    else:
        to_delete = pd.DataFrame(to_delete)
        to_keep = pd.DataFrame(to_keep)

    print('Found {0} local files on the server.'.format(to_delete.shape[0]))
    from datetime import datetime, timedelta
    deleteidx = []
    for i,(_,f) in enumerate(to_delete.iterrows()):
        mtime = datetime.fromtimestamp(f.mtime)
        now = datetime.today()
        if (now-mtime) > timedelta(weeks=keep_recent_weeks):
            deleteidx.append(i)
    print('Deleting {0} local files.'.format(len(deleteidx)))
    deleted = []
    for i in tqdm(deleteidx):
        if not dry_run:
            try:
                os.remove(to_delete.iloc[i].filepath)
                deleted.append(to_delete.iloc[i])
            except Exception as err:
                print(err)
    if len(deleted):
        print('Deleted {0} local files.'.format(len(deleted)))
    deleted = pd.DataFrame(deleted)
    if to_keep.shape[0]>0:
        print('{0} files were kept.'.format(to_keep.shape[0]))
    # TODO: this recursive is a bit of a hack
    for i in range(5):
        for f in list(os.walk(labdata_preferences['paths'][0]))[1:]:
            if not f[2] and not f[1]:
                try:
                    os.rmdir(f[0])
                except:
                    pass
                finally:
                    print('Removing {0}'.format(f[0]))
    return deleted,to_delete,to_keep # Deleted, On the server, Not on server
