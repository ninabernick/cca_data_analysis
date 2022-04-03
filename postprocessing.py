import xarray as xr
import sys

#path to results file containing datasets
dirname = sys.argv[1]
ds = xr.open_mfdataset(dirname+'/*.nc')
ds_mean = ds.sel(mode=0).mean('random_state').mean('rep').mean('n_pc_skip').mean('x_feature').mean('y_feature')
ds_mean.to_netcdf(dirname + "_ds_merged.nc")
