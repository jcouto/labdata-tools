import argparse
from .rclone import *
import sys

class CLI_parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description = 'labdata - tools to handle data from shared remotes',
            usage = ''' labdata <command> [<args>]

The commands are: 
    subjects                            list subjects in the remote
    sessions <subject_name>             list sessions for a subject
    get                                 get a dataset
    upload  <subject_name (optional)>   uploads a dataset
''')
        parser.add_argument('command', help= 'type: labdata <command> -h for help')

        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('The command [{0}] was not recognized. '.format(args.command))
            parser.print_help()
            exit(1)
        getattr(self,args.command)()

    def upload(self):
        parser = argparse.ArgumentParser(
            description = 'Upload datafolder in "path" to the remote.',
            usage = 'labdata upload <subject_name (optional)>')
        
        parser.add_argument('subject', action='store', default='',
                            type=str, nargs = '?')
        parser.add_argument('-l','--bwlimit', action='store', default=None, type=int)
        parser.add_argument('--path-index', action='store',
                            default=0, type=int)
        parser.add_argument('--overwrite', action='store_true',
                            default=False)
        args = parser.parse_args(sys.argv[2:])
        rclone_upload_data(subject = args.subject,
                           path_idx = args.path_index,
                           bwlimit = args.bwlimit,
                           overwrite = args.overwrite)
        
    def subjects(self):
        parser = argparse.ArgumentParser(description = 'list subjects in the database',
                                         usage = 'labdata subjects')

        subjects = rclone_list_subjects()

        [print(s,flush=True) for s in subjects]

    def sessions(self):
        parser = argparse.ArgumentParser(
            description = 'list sessions for subjects in the database',
            usage = 'labdata sessions <subject_name>')
        
        parser.add_argument('subject', action='store', default='', type=str)
        parser.add_argument('-i','--includes', action='store', default=[], type=str, nargs='+')
        parser.add_argument('-e','--excludes', action='store', default=[], type=str, nargs='+')
        
        args = parser.parse_args(sys.argv[2:])
        inc = args.includes
        #for i,v in enumerate(inc):
        #    if not v.startswith('*'):
        #        inc[i] =  '*' + v
        #    if not v.endswith('*'):
        #        inc[i] +=  '*'
        files = rclone_list_files(subject = args.subject,
                                  filters = inc)
        if not len(files):
            print('Found no sessions.')
            return
        for ses in files.session.unique():
            print(ses,flush=True)
            t = files[files.session == ses]
            print('\t'+'\n\t'.join(t.datatype.unique()), flush=True)

    def get(self):
        parser = argparse.ArgumentParser(
            description = 'fetch data from the database',
            usage = 'labdata get <subject_name>')
        
        parser.add_argument('-a','--subject', action='store', default=[], type=str,nargs='+')
        parser.add_argument('-s','--session', action='store', default=[], type=str,nargs='+')
        parser.add_argument('-d','--datatype', action='store', default=[], type=str,nargs='+')
        parser.add_argument('-i','--includes', action='store', default=[], type=str, nargs='+')
        parser.add_argument('-e','--excludes', action='store', default=[], type=str, nargs='+')
        
        args = parser.parse_args(sys.argv[2:])
        inc = args.includes
        for subject in args.subject:
            for session in args.session:
                for datatype in args.datatype:
                    rclone_get_data(subject = subject,
                                    session = session,
                                    datatype = datatype,
                                    includes = args.includes,
                                    excludes = args.excludes)
        
def main():
    CLI_parser()

