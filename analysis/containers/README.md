## Singularity containers

### Building a container

``singularity build --fakeroot container_spks.sif container_spks.def``

### Launching an interactive container

``singularity shell --nv -B /scratch,/home/data container_kilosort25.sif``

*Note:* labdata `analysis` can be linked with the $LABDATA_PREFERENCES variable


