# Hoffman2 setup
Labdatatools can be run from any submission-based cluster with proper configuration. Currently supports job submission via slurm and univa grid engine (fork of sun grid engine). 
## General user setup
Labdatatools, analysis software (kilsort, suite2p, etc.), and required conda environments have all been intalled on Hoffman2, requiring minimal configuration for each user.

All shared software can be found in 
```
/u/project/churchland/apps
```
This is the shared project folder where all Churchland lab software lives. You should also find your personal project folder under `/u/project/churchland/`. This is seperate from your `$HOME` folder and has a larger capacity, so you may want to configure your labdatatools config file to put data there.
## Dev setup


