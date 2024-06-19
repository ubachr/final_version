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