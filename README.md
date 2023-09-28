# labdata-tools
Utilities to copy data to and from the data server through python using rclone.
Run custom pipelines from defined data locations. 

### Command line:

#### Delete local data

``labdata clean_local -w 4``

Deletes all data from the local directory that is the same on the remote and is older than 4 weeks.

``labdata clean_local -n``

Skips the checksum, compares only filesize.

``labdata clean_local JC027``

Deletes all data from a specific subject.

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

To list the available commands: ``labdata --help``
### Installation

Clone the repository to a folder in your computer and do:

``python setup.py install`` or ``develop`` to if you plan modifying the code.

#### Hoffman2 setup (Churchland Lab members)

If you would like to use labdatatools on the UCLA cluster (Hoffman2), please see HOFFMAN_SETUP.md for instructions.

To run suite2p on Hoffman2 run:

``labdata submit suite2p --queue highp --module anaconda3 --conda-env suite2p -a JC081 -s 20220526_201159 --no-upload -- --file-filter JC081_20220526_201159_000_000``