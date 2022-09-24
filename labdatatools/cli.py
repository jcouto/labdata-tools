import argparse
from .rclone import *
from .analysis import load_plugins
import sys
from .slurm import has_slurm
from .uge import has_uge
from .remote import submit_remote_job

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
    clean_local                         deletes old files from the local computer

    run <ANALYSIS> -a <SUBJECT> -s <SESSION>     Runs an analysis script
    submit <ANALYSIS> -a <SUBJECT> -s <SESSION>   Submits an analysis script the queue (currently supports slurm or univa grid engine)
            
''')
        parser.add_argument('command', help= 'type: labdata <command> -h for help')

        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('The command [{0}] was not recognized. '.format(args.command))
            parser.print_help()
            exit(1)
        getattr(self,args.command)()

    def submit(self):
        parser = argparse.ArgumentParser(
            description = 'Process a dataset locally using slurm or univa grid engine.',
            usage = 'labdata submit <ANALYSIS> -- <PARAMETERS>')
        
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
        parser.add_argument('--conda-env', action='store',default = None, type=str)
        parser.add_argument('--module', action='store',default = None, type=str)
        parser.add_argument('-t','--walltime', action='store',default = None, type=str)
        sysargs = sys.argv[2:]
        analysisargs = []
        if '--' in sys.argv:
            sysargs = sys.argv[2:sys.argv.index('--')]
            analysisargs = sys.argv[sys.argv.index('--'):]
        args = parser.parse_args(sysargs)
        plugins = load_plugins()
        if has_slurm():
            if args.list_queues:
                os.system('sinfo')
                return
            if args.list_jobs:
                os.system('squeue')
                return
        elif has_uge():
            if args.list_queues:
                os.system('qhost')
                return
            if args.list_jobs:
                os.system('qstat -u echo $USER')
                return
        else:
            print('No batch submission cluster detected on the local computer.')
            labdatacmd = ' '.join(['labdata'] + sys.argv[1:])
            subject = None
            session = None
            if not args.subject == ['']:
                subject = args.subject[0]
            if not args.session == ['']:
                session = args.session[0]

            submit_remote_job(labdatacmd,subject=subject,session = session)
            return

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
        if analysis.has_gui:
            print('This command needs to be ran interactively, use "run" instead.')
        jobnumber = analysis.submit(analysisargs,
                                   conda_environment = args.conda_env,
                                   ncpuspertask = args.ncpus,
                                   memory=args.memory,
                                   walltime=args.walltime,
                                   partition=args.queue,
                                   module=args.module)
        print('Job submitted {0}'.format(jobnumber))

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
        parser.add_argument('--partial',
                            action='store', default=None, type=str)

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
                upload = args.no_upload,
                partial_run = args.partial)
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
                             '.ipynb_checkpoints',
                             '._.DS_Store',
                             '.DS_Store',
                             'dummy',
                             'FakeSubject'], type=str, nargs='+')
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
            if args.session==[''] and args.datatype!=['']: #handle case of empty sessions but datatype is specified
                data = rclone_list_files(subject)
            else:
                data = None
            for datatype in args.datatype:
                if data is not None:
                    args.session = data.session[data.datatype==datatype].unique().tolist() #slicing DataFrame to get sessiondates that contain desired datatype
                for session in args.session:
                    rclone_get_data(subject = subject,
                                    session = session,
                                    datatype = datatype,
                                    includes = args.includes,
                                    excludes = args.excludes)
    def clean_local(self):
        parser = argparse.ArgumentParser(
            description = 'fetch data from the database',
            usage = 'labdata clean_local --subject <SUBJECT> --no-checksum --keep-recent-weeks 5 --dry-run')

        # todo: add stuff to select only some animals
        parser.add_argument('-s','--subject', action='store', default=None, type=str)
        parser.add_argument('-e','--except', action='store',
                            default=['dummy','FakeSubject'], type=str,nargs='+')
        parser.add_argument('-n','--no-checksum', action='store_false',
                            help = 'skip the checksum', default=True)
        parser.add_argument('-w','--keep-recent-weeks', action='store',
                            type = int, default=5,
                            help='Number of weeks to keep data for.')
        parser.add_argument('--dry-run', action='store_true', default=False)
        
        
        args = parser.parse_args(sys.argv[2:])        
        from .cleanup import clean_local_files
        clean_local_files(subject = args.subject,
                          checksum = args.no_checksum,
                          dry_run = args.dry_run,
                          keep_recent_weeks = args.keep_recent_weeks)
                
def main():
    CLI_parser()

