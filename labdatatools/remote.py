from .utils import *
import sys

def get_paramiko_client():
    if 'remote_queue' in labdata_preferences.keys():
        try:
            import paramiko
        except:
            print('"pip install paramiko" to install remote submissions.')
            sys.exit()
        for required in ['remote','user']:
            if not required in labdata_preferences['remote_queue'].keys():
                print('There is no "{0}" key in the "remote_queue" preferences.'.format(required))
                sys.exit()
    remotehost = labdata_preferences['remote_queue']['remote']
    remoteuser = labdata_preferences['remote_queue']['user']
    remotepass = None
    if 'password' in labdata_preferences['remote_queue'].keys():
        remotepass = labdata_preferences['remote_queue']['password']
    if remotepass is None:
        try:
            from PyQt5.QtWidgets import QApplication, QInputDialog, QLineEdit
            app = QApplication([])
            text, ok = QInputDialog.getText(
                None, "labdatatools remote machine password",
                "Password for user {0} on {1}".format(remoteuser,remotehost),
                QLineEdit.Password)
            if ok:
                remotepass = text
        except:
            # use the cli for the password
            import getpass
            remotepass = getpass.getpass(prompt="Password for {}: ".format(remotehost))

    client = paramiko.client.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(remotehost, username=remoteuser, password=remotepass)
    return client

def submit_remote_job(labdatacmd, subject = None, session = None):
    client = get_paramiko_client()
    if not session is None and not subject is None:
        print('Checking if upload is needed.')
        from .rclone import rclone_upload_data
        rclone_upload_data(subject=subject,
                           session = session)
    # submit to remote computer
    # _stdin, _stdout,_stderr = client.exec_command(labdatacmd)
    _stdin, _stdout,_stderr = client.exec_command('source ~/.bash_profile && {}'.format(labdatacmd))
    _, _stdout2, _ = client.exec_command('hostname')
    remotehost = _stdout2.read().decode()[:-1] # Escape new line
    print('\n\n[{0}] Running: {1} \n'.format(remotehost,labdatacmd))
    print(_stdout.read().decode())
    print(_stderr.read().decode())
    client.close()

def list_remote_jobs():
    client = get_paramiko_client()
    _stdin, _stdout,_stderr = client.exec_command('source ~/.bash_profile && labdata submit --list-jobs') #Recursively call on remote
    print(_stdout.read().decode())
    print(_stderr.read().decode()) #for testing, need to return one of these 
    _stdin.close() 
    client.close()
    
def list_remote_queues():
    client = get_paramiko_client()
    _stdin, _stdout,_stderr = client.exec_command('source ~/.bash_profile && labdata submit --list-queues')
    print(_stdout)
    print(_stderr) #for testing
    _stdin.close()
    client.close()


