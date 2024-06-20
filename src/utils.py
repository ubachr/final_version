#import required libraries
import geopandas as gpd
import xarray as xr
import pandas as pd
import os, glob
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.colors
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib.cm as cm

import rioxarray as rio
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window, transform as rio_window_transform
from rasterio.windows import from_bounds
from rasterio.transform import Affine
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.features import rasterize
from shapely.geometry import box

import dask
import dask.array as da
from dask.delayed import delayed

from osgeo import gdal, ogr, osr


def rasterize_vector_to_raster(
    vector_data, 
    field,
    output_raster_path, 
    raster_width, 
    raster_height, 
    minx,  # x-coordinate of the upper-left corner (west)
    maxy,  # y-coordinate of the upper-left corner (north)
    resolution=10, 
    raster_crs='EPSG:3035', 
    chunk_size=1000, 
    compression='lzw'
):
            
    """
    Rasterize a vector dataset based on specific field to a raster file with specified properties.

    Parameters:
    - vector_path: str, path to the vector file (e.g., shapefile)
    - field: str, name of field to burn the raster (e.g., 'id')
    - output_raster_path: str, path to the output raster file
    - raster_width: int, width of the output raster
    - raster_height: int, height of the output raster
    - minx: float, x-coordinate of the upper-left corner of the raster (origin, west)
    - maxy: float, y-coordinate of the upper-left corner of the raster (origin, north)
    - resolution: int, pixel size in meters (default is 10 meters)
    - raster_crs: str, coordinate reference system for the output raster
    - chunk_size: int, size of chunks for processing
    - compression: str, compression type for the output raster
    """
     
    # Ensure vector data has a CRS
    if vector_data.crs is None:
        raise ValueError("Input vector data does not have a CRS defined.")
    
    # Define the transform for the output raster
    raster_transform = from_origin(minx, maxy, resolution, resolution)

    def process_chunk(window, transform, vector_data):
        local_transform = rio_window_transform(window, transform)
        bbox = rasterio.windows.bounds(window, transform=transform)
        bbox_geom = box(*bbox)
        vector_chunk = vector_data[vector_data.geometry.intersects(bbox_geom)].copy()

        if vector_chunk.empty:
            return np.zeros((window.height, window.width), dtype='uint32')

        rasterized_chunk = rasterize(
            ((geom, value) for geom, value in zip(vector_chunk.geometry, vector_chunk[field])),
            out_shape=(window.height, window.width),
            transform=local_transform,
            fill=0,
            dtype='uint32'
        )

        return rasterized_chunk

    tasks = []

    with rasterio.open(
        output_raster_path,
        'w',
        driver='GTiff',
        height=raster_height,
        width=raster_width,
        count=1,
        dtype='uint32',
        crs=raster_crs,
        transform=raster_transform,
        compress=compression
    ) as dst:
        for i in range(0, raster_height, chunk_size):
            for j in range(0, raster_width, chunk_size):
                # Adjust chunk size for edges
                window_width = min(chunk_size, raster_width - j)
                window_height = min(chunk_size, raster_height - i)
                window = Window(j, i, window_width, window_height)
                
                task = delayed(process_chunk)(window, raster_transform, vector_data)
                tasks.append(task)

        results = dask.compute(*tasks)

        for (result, (row_off, col_off)) in zip(results, [(i, j) for i in range(0, raster_height, chunk_size) for j in range(0, raster_width, chunk_size)]):
            # Adjust chunk size for edges
            window_width = min(chunk_size, raster_width - col_off)
            window_height = min(chunk_size, raster_height - row_off)
            window = Window(col_off, row_off, window_width, window_height)
            
            dst.write(result, window=window, indexes=1)

def rescale_and_clip_to_target(input_raster_path, target_raster_path, output_raster_path, compression):
    """
    Rescale a raster to match the resolution and clip it to the extent of another raster.

    Parameters:
    - input_raster_path: str, path to the input raster file (e.g., 'path/to/input_raster.tif')
    - target_raster_path: str, path to the target raster file (e.g., 'path/to/target_raster.tif')
    - output_raster_path: str, path to the output raster file (e.g., 'path/to/output_raster.tif')
    - compression: str, compression type (e.g., 'zstd', 'deflate', 'lwz')
    """
    
    with rasterio.open(target_raster_path) as target_src:
        # Get the extent and resolution of the target raster
        target_extent = target_src.bounds
        target_resolution = target_src.res[0]  # Assuming square pixels
    
    with rasterio.open(input_raster_path) as src:
        # Ensure the data type is uint16
        if src.dtypes[0] != 'int16':
            raise ValueError("Input raster is not of type int16")
        
        # Calculate the new transform and dimensions for the target extent
        new_transform = Affine(target_resolution, 0, target_extent.left, 0, -target_resolution, target_extent.top)
        new_width = int((target_extent.right - target_extent.left) / target_resolution)
        new_height = int((target_extent.top - target_extent.bottom) / target_resolution)
        
        # Read the data, resampling and clipping as necessary
        window = from_bounds(*target_extent, transform=src.transform)
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            window=window,
            resampling=Resampling.nearest
        )
        
        # Update metadata
        profile = src.profile
        profile.update({
            'transform': new_transform,
            'width': new_width,
            'height': new_height,
            'res': (target_resolution, target_resolution),
            'dtype': 'int16',
            'compress': f'{compression}'
        })
        
        # Write the resampled and clipped data to a new file
        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            dst.write(data.astype('int16'))


