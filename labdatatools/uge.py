from .utils import *
import subprocess as sub

LABDATA_UGE_FOLDER = pjoin(os.path.expanduser('~'),'labdatatools','uge') #TODO: get this from preferences file

def has_uge():
    proc = sub.Popen('qstat', shell=True, stdout=sub.PIPE, stderr = sub.PIPE)
    out,err = proc.communicate()
    if len(err):
            return False
    return True

def submit_uge_job(jobname,
                   command,
                   ntasks=None,
                   ncpuspertask = None,
                   memory=None,
                   walltime=None,
                   partition=None,
                   conda_environment=None,
                   module_environment=None,
                   mail=None,
                   sbatch_append='',
                   **kwargs):

    if ncpuspertask is None and ntasks is None:
        ncpuspertask = 1
        ntasks = 1
    if ntasks is None:
        ntasks = 1
    if ncpuspertask is None:
        ncpuspertask = 1
    if partition == 'gpu' and module_environment is None:
        raise Exception('If you would like to use a GPU node, you need to specify a CUDA module to activate.')
# TODO: Implement ntasks below
    #from datetime import datetime
    #rundate = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
    ugejobfile = '''#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o {logfolder}/{jobname}_$JOB_ID.stdout
#$ -j y
'''.format(jobname=jobname,
           logfolder=LABDATA_UGE_FOLDER,
           ntasks=ntasks,
           ncpus=ncpuspertask)
    ugejobfile += '#$ -l '
    if walltime is not None:
        ugejobfile += 'h_rt={},'.format(walltime)
    if memory is not None:
        ugejobfile += 'h_data={}G,'.format(int(memory/ncpuspertask))
        ugejobfile += 'h_vmem={}G,'.format(memory)
    if partition is not None:
        ugejobfile += '{}'.format(partition)
    ugejobfile += ' \n'
    if ncpuspertask is not None:
        ugejobfile += '#$ -pe shared {}\n'.format(ncpuspertask)
    if mail is not None:
        ugejobfile += '#$ -M {}\n#$ -m bea\n'.format(mail)
    ugejobfile += '\n. /u/local/Modules/default/init/modules.sh'
    ugejobfile += '\nsource $HOME/.bash_profile'
    if module_environment is not None:
        ugejobfile += '\nmodule load {}\n'.format(module_environment)
    if conda_environment is not None:
        ugejobfile += '\nconda activate {} \n'.format(conda_environment)

    ugejobfile += '\necho "JOB {} started on:  " `hostname -s` \n'.format(jobname)
    ugejobfile += 'echo "JOB {} started on:  " `date ` \n'.format(jobname)
    ugejobfile += 'echo " " \n'
    
    ugejobfile += 'echo "{}"\n'.format(command)
    ugejobfile += '{}'.format(command)

    ugejobfile += '\necho "JOB {} ended on:  " `hostname -s` \n'.format(jobname)
    ugejobfile += 'echo "JOB {} ended on:  " `date ` \n'.format(jobname)
    ugejobfile += 'echo " " \n'

    if not os.path.exists(LABDATA_UGE_FOLDER):
        os.makedirs(LABDATA_UGE_FOLDER)
    nfiles = len(glob(pjoin(LABDATA_UGE_FOLDER,'*.sh')))
    filename = pjoin(LABDATA_UGE_FOLDER,'{jobname}_{nfiles}.sh'.format(jobname=jobname,nfiles=nfiles+1))
    #filename = pjoin(LABDATA_UGE_FOLDER,'{jobname}_{rundate}.sh'.format(jobname=jobname,rundate=rundate))

    with open(filename,'w') as f:
        f.write(ugejobfile)
    folder,fname = os.path.split(filename)
    submit_cmd = 'cd {0} && qsub {2} {1}'.format(folder,fname,sbatch_append)
    proc = sub.Popen(submit_cmd, shell=True, stdout=sub.PIPE)
    out,err = proc.communicate()

    if b'Your job' in out:
        jobid = int(re.findall("Your job ([0-9]+)",str(out))[0])
        return jobid
    else:
        return None
