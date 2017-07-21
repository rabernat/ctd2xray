import numpy as np
from scipy.interpolate import interp1d
import functools
import os
import xarray as xr
import pandas as pd

def open_cchdo_as_mfdataset(paths, target_pressure,
                            pressure_coord='pressure',
                            concat_dim='time'):
    """Open cchdo hydrographic data in netCDF format, interpolate to
    specified pressures, and combine as an xarray dataset
    
    Parameters
    ----------
    paths : str or sequence
        Either a string glob in the form "path/to/my/files/*.nc" or an explicit
        list of files to open.
    target_pressure : arraylike
        Target pressure to which all casts are interpolated
    pressure_coord : str
        Name of the coordinate variable for pressure
    concat_dim : str
        Name of the dimension along which to concatenate casts
        
    Returns
    -------
    ds : xarray Dataset
    """
   
    # add time if missing
    timefun = _maybe_add_time_coord
    # create interpolation function for pressure
    interpfun = functools.partial(interp_coordinate,
                interp_coord=pressure_coord, interp_data=target_pressure)
    # create renaming function for concatenation
    renamefun = functools.partial(rename_0d_coords, new_dim=concat_dim)
    # compose together
    ppfun = compose(interpfun, renamefun, timefun)
    #paths = os.path.join(ddir, match_pattern)
    return xr.open_mfdataset(paths, concat_dim=concat_dim, preprocess=ppfun)

def interp_coordinate(ds,
        interp_coord, interp_data, drop_original=True,
        interp_suffix = '_i',
        interp_kwargs={'bounds_error': False}):
    """Interpolate xarray dataset to a new coordinate using
    ``scipy.interpolate.interp1d``
    
    Paramters
    ---------
    interp_coord : str
        The name of the coordinate along which to perform interpolation
    interp_data : arraylike
        New data points to which to interpolate
    drop_original : bool
        If ``True``, original coordinate and uninterpolated data variables
        are dropped from the output dataset
    interp_suffix : str
        The suffix to add to interpolated coodinate and variable names
    interp_kwargs : dict
        Additional arguments to pass to ``scipy.interpolate.interp1d``
        
    Returns
    -------
    ds : xarray Dataset
        New dataset with interpolated variables
    """

    x = ds[interp_coord]
    xnew_name = x.name + interp_suffix

    for yname in ds.data_vars:
        y = ds[yname]
        dims = list(y.dims)
        if x.name in y.dims:
            axis = y.get_axis_num(x.name)
            i = interp1d(x.values, y.values, axis=axis, **interp_kwargs)
            # set up new dims and coords
            dims[axis] = xnew_name
            coords = {}
            for d in dims:
                if d==xnew_name:
                    coords[d] = interp_data
                else:
                    coords[d] = y.coords[d]

            ynew = xr.DataArray(i(interp_data), dims=dims, coords=coords)
            ds[y.name + interp_suffix] = ynew
            if drop_original:
                ds = ds.drop(y.name)
    if drop_original:
        ds = ds.drop(x.name)
    return ds

def rename_0d_coords(ds, new_dim):
    """Assign all zero-dimensional coodinate variables in an xarray dataset
    to be indexed by a different dimension
    
    Parameters
    ----------
    ds : xarray dataset
        The input dataset
    new_dim : str
        The dimension to which to assign all other zero-dimensional coords
        
    Returns
    -------
    new_ds : xarray dataset
        Output dataset with coordinates reassigned
    """
    for d in ds.dims:
        if (ds[d].ndim==1 and d!=new_dim):
            if len(ds[d]) == len(ds[new_dim]):
                # create new dataset with only new coordinate
                oldvar = ds[d]
                newvar = xr.DataArray(oldvar,
                            coords={new_dim: ds[new_dim].values},
                            dims=[new_dim,],
                            attrs=oldvar.attrs,
                            name=d)
                dsnew = newvar.to_dataset().set_coords(d)
                ds = ds.update(dsnew)
    return ds

def _maybe_add_time_coord(ds, attr_name='Cast_start_UTC', coord_name='time',
                          get_dim_from='station'):
    if coord_name in ds.dims:
        return ds
    else:
        newdims = ds[get_dim_from].dims
        assert len(newdims)==1
        return attribute_to_time_variable(ds, attr_name,
                variable_name=coord_name, variable_dim=newdims[0])

def attribute_to_time_variable(ds, attr_name,
                variable_name='time', variable_dim='time'):
    """Turn an attribute into a time coordinate.

    Parameters
    ----------
    ds : xarray dataset
    attr_name : str
        The dataset attribute to parse for time
    variable_name : str
        The new variable name for the time data
    variable_dim : str
        The dimension to use for the time data

    Returns
    -------
    ds_new : xarray dataset
    """
    time = np.array([pd.to_datetime(ds.attrs[attr_name], utc=True)])
    if variable_dim==variable_name:
        da = xr.DataArray(time, dims=[coord_name])
    else:
        da = xr.DataArray(time, coords={variable_dim:ds[variable_dim]})
    ds[variable_name] = da
    return ds

def compose(*functions):
    """Utility for composing multiple functions into one.
    """
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


