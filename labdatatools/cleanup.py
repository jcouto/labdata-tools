from .rclone import rclone_list_files
from .utils import list_files


def clean_local_files(subject = None, checksum = True, dry_run = False, keep_recent_weeks = 5):
    to_delete = []
    to_keep = []
    subjects = list_subjects()
    if not subject is None:
        subjects = subjects[subjects.subject == subject]
    for i,s in subjects.iterrows():
        localfiles = list_files(s.subject)
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
            remotefiles = rclone_list_files(s.subject) 
            for i,l in localfiles.iterrows():
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
    for i,f in to_delete.iterrows():
        mtime = datetime.fromtimestamp(f.mtime)
        now = datetime.today()
        if (now-mtime) > timedelta(weeks=keep_recent_weeks):
            deleteidx.append(i)
    print('Deleting {0} local files.'.format(len(deleteidx)))
    deleted = []
    for i in tqdm(deleteidx):
        if not dry_run:
            os.remove(localfiles.filepath.loc[i])
        deleted.append(localfiles.loc[i])

    if len(deleted):
        print('Deleted {0} local files.'.format(len(deleted)))
    deleted = pd.DataFrame(deleted)
    if to_keep.shape[0]>0:
        print('{0} files were kept.'.format(to_keep.shape[0]))
    # TODO: check empty folders and delete them
    return deleted,to_delete,to_keep # Deleted, On the server, Not on server
