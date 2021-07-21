# labdata-tools
Utilities to copy data to and from the data server through python using rclone.


## Command line usage:

### List sessions with suite2p and two_photon data:

``labdata sessions JC027 -i suite2p two_photon``

### Get all matlab files for a specific animal:

``labdata get JC027 -i "*.mat"

Or for datafiles with a specific name:

``labdata get -a cy11 -i *TheMudSkipper2*.mat``

### List subjects in the database:

``labdata subjects``


## Tutorial

Look at the ``examples`` folder.


## Instalation

Clone the repository to a folder in your computer and do:

``python setup.py install`` or ``develop`` to if you plan modifying the code.
