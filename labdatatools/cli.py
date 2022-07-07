import argparse
from .rclone import *
from .analysis import load_plugins
import sys

class CLI_parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description = 'labdata - tools to handle data from shared remotes',
            usage = ''' labdata <command> [<args>]

The commands are:
            
    subjects                            list subjects in the remote
    sessions  <SUBJECT>                 list sessions for a subject
    get <SUBJECT> -s <SESSION>          get a dataset

    upload <SUBJECT (optional)>         uploads a dataset

    run <ANALYSIS> -a <SUBJECT> -s <SESSION>     Runs an analysis script
    slurm <ANALYSIS> -a <SUBJECT> -s <SESSION>   Submits an analysis script the queue
            
''')
        parser.add_argument('command', help= 'type: labdata <command> -h for help')

        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('The command [{0}] was not recognized. '.format(args.command))
            parser.print_help()
            exit(1)
        getattr(self,args.command)()

    def slurm(self):
        parser = argparse.ArgumentParser(
            description = 'Process a dataset locally using slurm.',
            usage = 'labdata slurm <ANALYSIS> -- <PARAMETERS>')
        
        parser.add_argument('analysis', action='store', default = '',
                            type=str,nargs = '?')
        parser.add_argument('-a','--subject',
                            action='store', default=[''], type=str,nargs='+')
        parser.add_argument('-s','--session',
                            action='store', default=[''], type=str,nargs='+')
        parser.add_argument('-d','--datatypes',
                            action='store', default=[''], type=str,nargs='+')
        parser.add_argument('-i','--includes',
                            action='store', default=[], type=str, nargs='+')
        parser.add_argument('-e','--excludes',
                            action='store', default=[], type=str, nargs='+')
        parser.add_argument('--overwrite', action='store_true',
                            default=False)
        parser.add_argument('--no-upload', action='store_false',
                            default=True)
        parser.add_argument('-q','--queue',
                            action='store', default=None, type=str)
        parser.add_argument('-m','--memory',
                            action='store', default=None, type=int)
        parser.add_argument('-n','--ncpus', action='store',default = None, type=int)
        parser.add_argument('--list-queues',action='store_true',default = False)
        parser.add_argument('--list-jobs',action='store_true',default = False)

        
        sysargs = sys.argv[2:]
        analysisargs = []
        if '--' in sys.argv:
            sysargs = sys.argv[2:sys.argv.index('--')]
            analysisargs = sys.argv[sys.argv.index('--'):]
        args = parser.parse_args(sysargs)
        if args.list_queues:
            os.system('sinfo')
        if args.list_jobs:
            os.system('squeue')
        plugins = load_plugins()
        if args.analysis in ['avail','available','list'] or not args.analysis in [p['name'] for p in plugins]:
            print('Available analysis [{0}]'.format(
                labdata_preferences['plugins_folder']))
            if len(plugins):
                for p in plugins:
                    print('\t'+p['name'])
            else:
                print('''
                Add plugins to the folder to start. 
                The repository has examples in the analysis folder.
                ''')    
            return 
        analysis = plugins[
            [p['name'] for p in plugins].index(args.analysis)]['object'](
                subject = args.subject,
                session = args.session,
                datatypes = args.datatypes,
                includes = args.includes,
                excludes = args.excludes,
                overwrite = args.overwrite,
                upload = args.no_upload)
        analysis.parse_arguments(analysisargs)
        analysis.validate_parameters()

        
    def run(self):
        parser = argparse.ArgumentParser(
            description = 'Run an analysis on a dataset.',
            usage = 'labdata run <ANALYSIS> -- <PARAMETERS>')
        parser.add_argument('analysis', action='store', default = '',
                            type=str,nargs = '?')
        parser.add_argument('-a','--subject',
                            action='store', default=[''], type=str,nargs='+')
        parser.add_argument('-s','--session',
                            action='store', default=[''], type=str,nargs='+')
        parser.add_argument('-d','--datatypes',
                            action='store', default=[''], type=str,nargs='+')
        parser.add_argument('-i','--includes',
                            action='store', default=[], type=str, nargs='+')
        parser.add_argument('-e','--excludes',
                            action='store', default=[], type=str, nargs='+')
        parser.add_argument('--overwrite', action='store_true',
                            default=False)
        parser.add_argument('--no-upload', action='store_false',
                            default=True)

        sysargs = sys.argv[2:]
        analysisargs = []
        if '--' in sys.argv:
            sysargs = sys.argv[2:sys.argv.index('--')]
            analysisargs = sys.argv[sys.argv.index('--'):]
        args = parser.parse_args(sysargs)
        plugins = load_plugins()
        if args.analysis in ['avail','available','list'] or not args.analysis in [p['name'] for p in plugins]:
            print('Available analysis [{0}]'.format(
                labdata_preferences['plugins_folder']))
            if len(plugins):
                for p in plugins:
                    print('\t'+p['name'])
            else:
                print('''
                Add plugins to the folder to start. 
                The repository has examples in the analysis folder.
                ''')
            return 

        analysis = plugins[
            [p['name'] for p in plugins].index(args.analysis)]['object'](
                subject = args.subject,
                session = args.session,
                datatypes = args.datatypes,
                includes = args.includes,
                excludes = args.excludes,
                overwrite = args.overwrite,
                upload = args.no_upload)
        analysis.parse_arguments(analysisargs)

        analysis.validate_parameters()
        analysis.process()

    def upload(self):
        parser = argparse.ArgumentParser(
            description = 'Upload datafolder in "path" to the remote.',
            usage = 'labdata upload <subject_name (optional)>')
        
        parser.add_argument('subject', action='store', default='',
                            type=str, nargs = '?')
        parser.add_argument('-s','--session', action='store', default=None, type=str)
        parser.add_argument('-d','--datatype', action='store', default=None, type=str)

        parser.add_argument('-l','--bwlimit', action='store', default=None, type=int)
        parser.add_argument('--path-index', action='store',
                            default=0, type=int)
        parser.add_argument('-e','--excludes', action='store', default=
                            ['.phy',
                             '._.DS_Store',
                             '.DS_Store'], type=str, nargs='+')
        parser.add_argument('--overwrite', action='store_true',
                            default=False)
        parser.add_argument
        args = parser.parse_args(sys.argv[2:])
        rclone_upload_data(subject = args.subject,
                           session = args.session,
                           datatype = args.datatype,
                           path_idx = args.path_index,
                           bwlimit = args.bwlimit,
                           overwrite = args.overwrite,
                           excludes = args.excludes)
        
    def subjects(self):
        parser = argparse.ArgumentParser(description = 'list subjects in the database',
                                         usage = 'labdata subjects')

        subjects = rclone_list_subjects()

        [print(s,flush=True) for s in subjects]

    def sessions(self):
        parser = argparse.ArgumentParser(
            description = 'list sessions for subjects in the database',
            usage = 'labdata sessions <subject_name>')
        
        parser.add_argument('subject', action='store', default=[''], type=str, nargs='*')
        parser.add_argument('-f','--filters', action='store', default=[], type=str, nargs='+')
        parser.add_argument('-i','--includes', action='store', default=[],
                            type=str, nargs='+')
        parser.add_argument('-e','--excludes', action='store', default=[],
                            type=str, nargs='+')        
        parser.add_argument('--files',action='store_true',default = False)
        
        
        args = parser.parse_args(sys.argv[2:])
        inc = args.includes
        #for i,v in enumerate(inc):
        #    if not v.startswith('*'):
        #        inc[i] =  '*' + v
        #    if not v.endswith('*'):
        #        inc[i] +=  '*'
        for sub in args.subject:
            files = rclone_list_files(subject = sub,
                                      filters = args.filters,
                                      includes = args.includes,
                                      excludes = args.excludes)
            if not len(files):
                print('Found no sessions.')
                continue
            for subject in natsorted(files.subject.unique()):
                nfiles = files[files.subject == subject]
                print(subject,flush=True)
                for ses in natsorted(nfiles.session.unique()):
                    print(' '+ses,flush=True)
                    t = nfiles[nfiles.session == ses]
                    for dtype in natsorted(t.datatype.unique()):
                        print(' \t'+dtype, flush=True)
                        d = t[t.datatype == dtype]
                        if args.files:
                            for i,f in d.iterrows():
                                print('\t\t{0}'.format(f.filename))
                            
    def get(self):
        parser = argparse.ArgumentParser(
            description = 'fetch data from the database',
            usage = 'labdata get <subject_name>')
        
        parser.add_argument('subject', action='store', default=[''], type=str,nargs='+')
        parser.add_argument('-s','--session', action='store', default=[''], type=str,nargs='+')
        parser.add_argument('-d','--datatype', action='store', default=[''], type=str,nargs='+')
        parser.add_argument('-i','--includes', action='store', default=[], type=str, nargs='+')
        parser.add_argument('-e','--excludes', action='store', default=[], type=str, nargs='+')
        
        args = parser.parse_args(sys.argv[2:])
        for subject in args.subject:
            for session in args.session:
                for datatype in args.datatype:
                    rclone_get_data(subject = subject,
                                    session = session,
                                    datatype = datatype,
                                    includes = args.includes,
                                    excludes = args.excludes)
    def clean_local(self):
        parser = argparse.ArgumentParser(
            description = 'fetch data from the database',
            usage = 'labdata clean_local')

        # todo: add stuff to select only some animals
        parser.add_argument('-e','--except', action='store', default=[], type=str,nargs='+')
        
        args = parser.parse_args(sys.argv[2:])        

        
        
def main():
    CLI_parser()

