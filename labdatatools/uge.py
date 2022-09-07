from .utils import *

LABDATA_UGE_FOLDER = pjoin(os.path.expanduser('~'),'labdatatools','uge')

# Todo: implement distributed submission for jobs that can be run in a distributed memory fashion.
# For now, we will default to shared memory submisison on one node.

def submit_uge_job(jobname,
                   command,
                   privatenode=False,
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
        from multiprocessing import cpu_count
        ncpuspertask = cpu_count()
        ntasks = 1
    if ntasks is None:
        ntasks = 1
    if ncpuspertask is None:
        ncpuspertask = 1
    usegpu = False #todo: add line to check if desired analyis can use gpu nodes, or maybe define this elsewhere
    if usegpu and privatenode
        raise Exception('Private node does not have a GPU! GPU jobs must be submitted to public queue')



# Todo: need to load cuda module if using gpu
