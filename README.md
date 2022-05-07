## senior thesis project

## Components:
- Jupyter notebooks for analyzing and visualizing data
  - all .ipynb files
- jobs for running analyses on compute cluster
  - `c3a_job_spawner.py`, `analyze_model_job.py`, `postprocessing.py`, `processing_job.sh`
- data from analyses
  - `final_data_zarr`: results from standard C3A and CCA
  - `linear_c3a_results_zarr`: results from linear weighted C3A
  - `log_c3a_results_zarr`: results from log weighted C3A
  - `weighted_c3a_results_zarr`: results from other weighted C3A trials

## Workflow:
- Update desired parameter values in `c3a_job_spawner.py` and desired algorithms to run in `analyze_model_job.py`, then run `python c3a_job_spawner.py [desired results directory]`
  - you may be rate limited in the amount of jobs you can spawn (I believe ~200/hour). Each spawned instance of `analyze_model_job.py` is one job, so make sure the "outer parameters" in `c3a_job_spawner.py` do not result in too many jobs. Additionally, it is advised to use the *day* queue to get your jobs run immediately, but the maximum time allowed is 1 day, so do not make the inner parameter set too large. 
- Data will be stored in `\[results_directory\]/ds_*.nc`. To combine results into one `xarray.Dataset`, update the path to the results folder in `processing_job.sh` then run `sbatch processing_job.sh`.
  - Data will be written to a `zarr` instead of `NetCDF` to drastically reduce the amount of space needed to store it. This may lead to some issues with encodings if you try to open these files and then rewrite them.
- If you want to combine overall datasets, each with a different algorithm tested, use `combine_datasets.py`

I found that it was difficult to do any development or data analysis on the cluster due to connectivity issues with VSCode, so I used github to keep the code repo on the cluster updated and transferred data files to my personal machine to run analyses.


