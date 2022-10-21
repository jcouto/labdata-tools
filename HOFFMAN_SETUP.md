# Hoffman2 setup
Labdatatools can be run from any submission-based cluster with proper configuration. Currently supports job submission via slurm and univa grid engine (fork of sun grid engine). 

## Churchland Labmember Setup
#### Getting Hoffman2 Access
To get set up with Hoffman2 access, follow the documentation [here](https://www.hoffman2.idre.ucla.edu/Accounts/Requesting-an-account.html). The docs here have good general information on how to use the cluster, but it can be cumbersome. One of the goals of labdatatools is to abstract this away, making cluster submission more straightforward. Ask Max or Joao if you have any questions.

Labdatatools, analysis software (kilsort, suite2p, etc.), and required conda environments have all been intalled on Hoffman2, requiring minimal configuration for each user.

All shared software can be found in `/u/project/churchland/apps`
This is the shared project folder where all Churchland lab software lives. You should also find your personal project folder under `/u/project/churchland/`. This is seperate from your `$HOME` folder and has a larger capacity, so you may want to configure your labdatatools config file to put data there.

#### Configure Path and Anaconda
We will first need to configure the user Path and Anaconda (preinstalled on Hoffman) to find the required analysis environments and binaries.
Run the following commands once signed into a Hoffman login node. This only needs to be done once for initial configuration.
```
echo 'export PATH="/u/project/churchland/apps/bin:$PATH"' >> ~/.bashrc #add churchland binaries to path
module load anaconda3 #load anaconda module
conda config --add envs_dirs /u/project/churchland/apps/condaenvs #add our shared environments to anaconda search path
conda config --set env_prompt '({name})'#shorter environment names
conda init #reinitialize conda
echo 'module load anaconda3' >> ~/.bash_profile #load anaconda by default next time
echo 'conda activate churchlandenv' >> ~/.bash_profile #activate churchlandenv by default 
```
Now exit the terminal and sign back in to Hoffman. You should see `(churchlandenv)` on the left side of the command line. 

#### Configure rclone
Run `rclone config` and follow the instructions for configuring a Google Team Drive. Be sure to give full read/write access. Default options are all fine, except you will want to select `no` when asked if you would like to use autoconfig (because you are on a remote machine).
You can name the shared drive whatever you want, but we recommend `churchland_data`, because this is the default setting specified in the labdatatools config file. If you choose something different it will have to be changed there.

Now type and run `labdata sessions JC081`! If you see a list of sessions for this animal then labdata is working and you should be able to directly run labdata upon your next sign in.

#### Hoffman2 important locations
Each user has their own config folder found in `$HOME\labdatatools`. This folder also contains analysis plugins and the output of batch submission jobs. You can edit these, or create new analysis plugins, without affecting how labdata runs for other users.

Within the labdata config file, we recommend editing the `"paths"` field to point to your project folder: `/u/project/churchland/$USER`. Labdatatools is set by default to put data in your `$HOME` folder, but Hoffman limits users to 40G of storage there. We have a combined 6T of disk-based storage in our project folder. Each user also has a personal folder `$SCRATCH` with 2T of flash storage. You could use this as your data folder, but data here is deleted after 14 days, so make sure you save the output of any analysis scripts back to the GDrive!

Please run `labdata clean_local` somewhat regularly so we don't run out of space. But be sure your results have been uploaded to the GDrive before cleaning these local files.

There is also a special directory `$TMPDIR`. This points to the local NVMe drive that is present on whatever compute node is running your analysis. Use this if you are creating an analysis plugin that is VERY I/O intensive. Do not save any long-term output here, it will be lost when your job finishes. See [this link](https://www.hoffman2.idre.ucla.edu/Using-H2/Storage/sgeenviron.py.html) with helpful info on making use of `$TMPDIR` in Python scripts.
#### Extras
If you are just wanting to run `labdata get <subject>` to download some data to Hoffman, this should be done from one of the dedicated [transfer nodes](https://www.hoffman2.idre.ucla.edu/Using-H2/Data-transfer.html). It can be done from a standard login node, but the rclone copy command may be unexpectedly terminated if the network utilization is too high. If you submit a job to the queue system, you do not have to worry about this. Any nonlocal data will be pulled via a compute node before the analysis is run.

## Dev setup
