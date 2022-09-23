def submit_remote_job(labdatacmd, subject = None, session = None):
    raise NotImplementedError('Need to merge submit_remote_slurm_job and submit_remote_uge_job')
    if 'remote_queue' in labdata_preferences.keys():
        try:
            import paramiko
        except:
            print('"pip install paramike" to install remote submissions.')
            sys.exit()
        for required in ['remote','user']:
            if not required in labdata_preferences['slurm'].keys():
                print('There is no "{0}" key in the "remote_queue" preferences.'.format(required))
                sys.exit()
    remotehost = labdata_preferenes['remote_queue']['remote']
    remoteuser = labdata_preferences['remote_queue']['user']
    remotepass = None
