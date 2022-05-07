import xarray as xr
import sys

#path to results file containing datasets
dirname = sys.argv[1]
#if there are any empty files or datasets with fewer than 28 vbls, open_mfdataset will fail
#preprocessing will print length of data variables so if it's less than 28 delete the file and try
#running this script again
def preprocess(ds):  
     print(ds.encoding['source'])
     print(len(ds.data_vars))
ds = xr.open_mfdataset(dirname+'/*.nc')
ds_mean = ds.sel(mode=0).mean('random_state').mean('rep').mean('n_pc_skip').mean('ax').mean('aya').mean('ayb').mean('x_feature').mean('y_feature')
ds_mean.to_zarr(dirname + "_ds_merged_zarr")