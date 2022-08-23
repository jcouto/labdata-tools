from .utils import *

LABDATA_SLURM_FOLDER = pjoin(os.path.expanduser('~'),'labdatatools','slurm')

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
    import subprocess as sub
    proc = sub.Popen(submit_cmd, shell=True, stdout=sub.PIPE)
    out,err = proc.communicate()
    
    if b'Submitted batch job' in out:
        jobid = int(re.findall("Submitted batch job ([0-9]+)", str(out))[0])
        return jobid
    else:
        return None
