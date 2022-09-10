from .utils import *
import subprocess as sub

LABDATA_SLURM_FOLDER = pjoin(os.path.expanduser('~'),'labdatatools','slurm')

def has_slurm():
    proc = sub.Popen('sinfo', shell=True, stdout=sub.PIPE, stderr = sub.PIPE)
    out,err = proc.communicate()
    if len(err):
        return False
    return True

def submit_slurm_job(jobname,
                     command,
                     ntasks=None,
                     ncpuspertask = None,
                     memory=None,
                     walltime=None,
                     partition=None,
                     conda_environment=None,
                     module_environment=None,
                     mail=None,
                     sbatch_append='',**kwargs):

    if ncpuspertask is None and ntasks is None:
        from multiprocessing import cpu_count
        ncpuspertask = cpu_count()
        ntasks = 1
    if ntasks is None:
        ntasks = 1
    if ncpuspertask is None:
        ncpuspertask = 1
        
    sjobfile = '''#!/bin/bash -login
#SBATCH --job-name={jobname}
#SBATCH --output={logfolder}/{jobname}_%j.stdout
#SBATCH --error={logfolder}/{jobname}_%j.stdout

#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={ncpus}
'''.format(jobname = jobname,
           logfolder = LABDATA_SLURM_FOLDER,
           ntasks = ntasks,
           ncpus = ncpuspertask)
    if not walltime is None:
        sjobfile += '#SBATCH --time={0} \n'.format(walltime)
    if not memory is None:
        sjobfile += '#SBATCH --mem={0} \n'.format(memory)
    if not partition is None:
        sjobfile += '#SBATCH --partition={0} \n'.format(partition)
    if not mail is None:
        sjobfile += '#SBATCH --mail-user={0} \n#SBATCH --mail-type=END,FAIL \n'.format(mail)
    if not module_environment is None:
        sjobfile += '\n module purge\n'
        sjobfile += '\n module load {0} \n'.format(module_environment)
    if not conda_environment is None:
        sjobfile += 'conda activate {0} \n'.format(conda_environment)
    sjobfile += '''echo JOB {jobname} STARTED `date`
{cmd}
echo JOB FINISHED `date`
'''.format(jobname = jobname, cmd = command)
    if not os.path.exists(LABDATA_SLURM_FOLDER):
        os.makedirs(LABDATA_SLURM_FOLDER)
    nfiles = len(glob(pjoin(LABDATA_SLURM_FOLDER,'*.stdout')))
    filename = pjoin(LABDATA_SLURM_FOLDER,'{jobname}_{nfiles}.sh'.format(jobname = jobname,
                                                                              nfiles = nfiles+1))
    with open(filename,'w') as f:
        f.write(sjobfile)
    folder,fname = os.path.split(filename)
    submit_cmd = 'cd {0} && sbatch {2} {1}'.format(folder,fname,sbatch_append)
    proc = sub.Popen(submit_cmd, shell=True, stdout=sub.PIPE)
    out,err = proc.communicate()
    
    if b'Submitted batch job' in out:
        jobid = int(re.findall("Submitted batch job ([0-9]+)", str(out))[0])
        return jobid
    else:
        return None


def submit_remote_slurm_job(labdatacmd, subject = None, session = None):
    if 'slurm' in labdata_preferences.keys():
        try:
            # required for ssh
            import paramiko
        except:
            print('"pip install paramiko" to install remote submissions.')
            sys.exit()
            # this is the remote computer key
        for required in ['remote','user']:
            if not required in labdata_preferences['slurm'].keys():
                print('There is no "{0}" key in the "slurm" preferences.'.format(required))
                sys.exit()
    remotehost = labdata_preferences['slurm']['remote']
    remoteuser = labdata_preferences['slurm']['user']
    remotepass = None
    if 'password' in labdata_preferences['slurm'].keys():
        remotepass = labdata_preferences['slurm']['password']
    if remotepass is None:
        try:
            from PyQt5.QtWidgets import QApplication, QInputDialog, QLineEdit
            app = QApplication([])
            text, ok = QInputDialog.getText(
                None, "labdatatools SLURM remote password",
                "Password for user {0} on {1}?".format(remoteuser,remotehost), 
                QLineEdit.Password)
            if ok:
                remotepass = text
        except:
            # use the cli for the password
            import getpass
            remotepass = getpass.getpass(prompt="SLURM remote host password?")
    # are data on google drive?
    if not session is None and not subject is None:
        print('Checking if upload is needed.')
        from .rclone import rclone_upload_data
        rclone_upload_data(subject=subject,
                           session = session)
    # submit to remote computer
    client = paramiko.client.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(remotehost, username=remoteuser, password=remotepass)
    _stdin, _stdout,_stderr = client.exec_command(labdatacmd)
    print('\n\n[{0}] Running: {1} \n'.format(remotehost,labdatacmd))
    print(_stdout.read().decode())
    print(_stderr.read().decode())
    client.close()
