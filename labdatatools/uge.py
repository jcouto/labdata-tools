from .utils import *

LABDATA_UGE_FOLDER = pjoin(os.path.expanduser('~'),'labdatatools','uge')

# TODO: implement distributed submission for jobs that can be run in a distributed memory fashion.
# For now, we will default to shared memory submisison on one node.
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
    ugejobfile='''#!/bin/bash
    #$ -cwd
    # error = Merged with joblog
    #$ -o {logfolder}/{jobname}_%j.stdout
    #$ -j y
    #$ -pe shared {ncpus}
    '''.format(jobname = jobname,
               logfolder = LABDATA_UGE_FOLDER,
               ntasks = ntasks,
               ncpus = ncpuspertask)
    ugejobfile += '#$ -l'
    if walltime is not None:
        ugejobfile += ' h_rt={},'.format(walltime)
    if memory is not None:
        ugejobfile += ' h_data={}G,'.format(memory)
    if partition is not None:
        ugejobfile += ' {}'.format(partition)
    ugejobfile += ' \n'
    if mail is not None:
        ugejobfile += '#$ -M {}\n#$ -m bea\n'.format(mail)
    if module_environment is not None:
        ugejobfile += '\nmodule purge'
        ugejobfile += '\nmodule load {}\n'.format(module_environment)
    if conda_environment is not None:
        ugejobfile += 'conda activate {} \n'.format(conda_environment)

    ugejobfile += '\necho "JOB {} started on:  " `hostname -s` \n'.format(jobname)
    ugejobfile += 'echo "JOB {} started on:  " `date ` \n'.format(jobname)
    ugejobfile += 'echo " " \n'

    ugejobfile += '{}'.format(command)

    ugejobfile += '\necho "JOB {} ended on:  " `hostname -s` \n'.format(jobname)
    ugejobfile += 'echo "JOB {} ended on:  " `date ` \n'.format(jobname)
    ugejobfile += 'echo " " \n'

    if not os.path.exists(LABDATA_UGE_FOLDER):
        os.makedirs(LABDATA_UGE_FOLDER)
    nfiles = len(glob(pjoin(LABDATA_UGE_FOLDER,'*.stdout')))
    filename = pjoin(LABDATA_UGE_FOLDER,'{jobname}_{nfiles}.sh'.format(jobname=jobname,nfiles=nfiles+1))

    with open(filename,'w') as f:
        f.write(ugejobfile)
    folder,fname = os.path.split(filename)
    submit_cmd = 'cd {0} && qsub {2} {1}'.format(folder,fname,sbatch_append)
    import subprocess as sub
    proc = sub.Popen(submit_cmd, shell=True, stdout=sub.PIPE)
    out,err = proc.communicate()

    if b'Submitted batch job' in out: #TODO: May need to modify this for UGE
        jobid = int(re.findall("Submitted batch job ([0-9]+)",str(out))[0])
        return jobid
    else:
        return None
