import xarray as xr

ds1 = xr.open_zarr('log_squared/resultsds_merged_zarr').sel(alg='c3a_log_squared')
ds2 = xr.open_zarr('linear_sqrt/resultsds_merged_zarr').sel(alg='c3a_linear_sqrt')
ds3 = xr.open_zarr('sqrt/resultsds_merged_zarr').sel(alg='c3a_sqrt')
ds4 = xr.open_zarr('log_c3a/trial1/log_c3a_results_zarr').sel(alg='c3a_log_weighted')
ds5 = xr.open_zarr('linear_c3a/trial1/linear_c3a_results_zarr').sel(alg='c3a_weighted')

ds_merged = xr.concat([ds1, ds2, ds3, ds4, ds5], dim='alg')
ds_merged['modality'] = ds_merged.modality.astype('str')
ds_merged.to_zarr('weighted_c3a_results_zarr', mode='w', safe_chunks=False) 