import cdsapi as cds
import xarray as xr

ds = xr.open_dataset("ERA5.grib", engine="netcdf4")

ds = xr.open_dataset("atmosphere.grib", engine="netcdf4")

