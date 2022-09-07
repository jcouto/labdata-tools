from .utils import *

LABDATA_UGE_FOLDER = pjoin(os.path.expanduser('~'),'labdatatools','uge')

def submit_uge_job(jobname,
                   command,
                   ntasks=None,
                   
