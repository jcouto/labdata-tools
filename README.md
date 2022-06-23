# labdata-tools
Utilities to copy data to and from the data server through python using rclone.

### Command line:

#### List sessions with suite2p or two_photon data:

``labdata sessions JC027 -f suite2p two_photon``

This takes a while if you don't specify subjects because it queries all files in the remote.

You can also list multiple subjects:
``labdata sessions JC027 JC066 JC065 -f suite2p two_photon``


Use the ``--files`` flag to list also the files.

#### Get all matlab files for a specific animal:

``labdata get JC027 -i "*.mat"``

Or for datafiles with a specific name:

``labdata get cy11 -i "*TheMudSkipper2*.mat"``

#### Get all files from 2 sessions:

``labdata get JC027 -s 20220220_000000 20220221_000000``

#### List subjects in the database:

``labdata subjects``

#### Upload data from the datapath directory

``labdata upload``

#### Upload data from a specific session

``labdata upload JC066 -s 20220126_172829``

### Tutorial

Look at the ``examples`` folder.

### Instalation

Clone the repository to a folder in your computer and do:

``python setup.py install`` or ``develop`` to if you plan modifying the code.
