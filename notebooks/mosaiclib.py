"""
Grid-based Memory-Efficient Mosaicking System
=============================================

This module creates memory-efficient mosaics by:
1. Creating a vector grid over the area of interest
2. Processing each grid cell independently 
3. Prioritizing images by temporal distance from target date
4. Supporting optional mask updates from coarser resolution data
5. Multiprocessing with joblib for parallel execution
6. Writing Cloud-Optimized GeoTIFFs (COGs)
7. Creating VRT for seamless viewing

Author: NASA CSDA Team
Date: 2026-01-02
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask as rasterio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.enums import Resampling as ResamplingEnum
from rasterio.shutil import copy as rio_copy
from rasterio.vrt import WarpedVRT
import os
from pathlib import Path
from datetime import datetime, timedelta
from shapely.geometry import box, Polygon
import warnings
from joblib import Parallel, delayed
from osgeo import gdal
import subprocess

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# GRID CREATION
# =============================================================================

def create_vector_grid(aoi_geom, grid_size_km=90, target_crs=None):
    """
    Create a vector grid over an area of interest.
    
    Parameters:
    -----------
    aoi_geom : shapely.geometry or geopandas.GeoDataFrame
        Area of interest geometry. Can be:
        - Shapely geometry (Polygon, MultiPolygon)
        - GeoDataFrame with one or more features
        - String 'alaska' for automatic Alaska extent
    grid_size_km : float, default=90
        Size of each grid cell in kilometers (creates square cells)
    target_crs : str, int, or CRS object, optional
        Target CRS for the grid. If None, automatically selects appropriate
        projection based on AOI centroid (UTM for lat < 60, Albers for Alaska/Arctic)
    
    Returns:
    --------
    geopandas.GeoDataFrame
        Grid with columns: ['tile_id', 'row', 'col', 'geometry']
    """
    
    # Handle special case for Alaska
    if isinstance(aoi_geom, str) and aoi_geom.lower() == 'alaska':
        # Alaska bounding box in WGS84
        # Approximately: -170 to -130 longitude, 51 to 72 latitude
        aoi_geom = box(-170, 51, -130, 72)
        aoi_gdf = gpd.GeoDataFrame({'geometry': [aoi_geom]}, crs='EPSG:4326')
        
        # Use Alaska Albers (EPSG:3338) for grid
        if target_crs is None:
            target_crs = 'EPSG:3338'
    
    # Convert to GeoDataFrame if needed
    elif not isinstance(aoi_geom, gpd.GeoDataFrame):
        aoi_gdf = gpd.GeoDataFrame({'geometry': [aoi_geom]}, crs='EPSG:4326')
    else:
        aoi_gdf = aoi_geom.copy()
    
    # Determine target CRS if not provided
    if target_crs is None:
        # Get centroid in WGS84
        centroid = aoi_gdf.to_crs('EPSG:4326').geometry.unary_union.centroid
        
        # For high latitudes (>60°), use appropriate Albers or Polar projection
        if abs(centroid.y) > 60:
            if centroid.y > 0:  # Northern hemisphere
                target_crs = 'EPSG:3338'  # Alaska Albers
            else:  # Southern hemisphere
                target_crs = 'EPSG:3031'  # Antarctic Polar Stereographic
        else:
            # Use UTM zone based on longitude
            utm_zone = int((centroid.x + 180) / 6) + 1
            if centroid.y >= 0:
                target_crs = f'EPSG:{32600 + utm_zone}'
            else:
                target_crs = f'EPSG:{32700 + utm_zone}'
    
    print(f"Creating grid in {target_crs}")
    
    # Reproject AOI to target CRS
    aoi_projected = aoi_gdf.to_crs(target_crs)
    
    # Get bounds in projected coordinates
    bounds = aoi_projected.total_bounds  # minx, miny, maxx, maxy
    
    # Convert grid size from km to meters
    grid_size_m = grid_size_km * 1000
    
    # Calculate grid dimensions
    xmin, ymin, xmax, ymax = bounds
    
    # Calculate number of columns and rows
    n_cols = int(np.ceil((xmax - xmin) / grid_size_m))
    n_rows = int(np.ceil((ymax - ymin) / grid_size_m))
    
    print(f"Grid dimensions: {n_rows} rows x {n_cols} columns ({n_rows * n_cols} total tiles)")
    
    # Create grid cells
    grid_cells = []
    tile_ids = []
    rows = []
    cols = []
    
    for row in range(n_rows):
        for col in range(n_cols):
            # Calculate cell bounds
            cell_xmin = xmin + (col * grid_size_m)
            cell_ymin = ymin + (row * grid_size_m)
            cell_xmax = cell_xmin + grid_size_m
            cell_ymax = cell_ymin + grid_size_m
            
            # Create cell geometry
            cell_geom = box(cell_xmin, cell_ymin, cell_xmax, cell_ymax)
            
            # Only include cells that intersect with AOI
            if cell_geom.intersects(aoi_projected.unary_union):
                grid_cells.append(cell_geom)
                tile_id = f"R{row:04d}C{col:04d}"
                tile_ids.append(tile_id)
                rows.append(row)
                cols.append(col)
    
    # Create GeoDataFrame
    grid_gdf = gpd.GeoDataFrame({
        'tile_id': tile_ids,
        'row': rows,
        'col': cols,
        'geometry': grid_cells
    }, crs=target_crs)
    
    print(f"Created {len(grid_gdf)} grid tiles intersecting AOI")
    
    return grid_gdf

# =============================================================================
# TEMPORAL PRIORITIZATION
# =============================================================================

def calculate_temporal_distance(image_date, target_date, target_doy=212):
    """
    Calculate temporal distance from target date and day of year.
    
    Parameters:
    -----------
    image_date : datetime.datetime
        Date of the image
    target_date : datetime.datetime
        Target year and date
    target_doy : int, default=212
        Target day of year (default 212 = July 31)
    
    Returns:
    --------
    float
        Temporal distance score (lower is better)
    """
    # Calculate year difference
    year_diff = abs(image_date.year - target_date.year)
    
    # Calculate day of year difference
    image_doy = image_date.timetuple().tm_yday
    doy_diff = abs(image_doy - target_doy)
    
    # Handle wrap-around (e.g., DOY 365 vs DOY 1)
    doy_diff = min(doy_diff, 365 - doy_diff)
    
    # Combined score: prioritize year, then DOY
    # Year difference is weighted much more heavily
    temporal_score = (year_diff * 365) + doy_diff
    
    return temporal_score

def prioritize_images_for_tile(footprints_subset, target_year, target_doy=212, 
                                delta_years=5, include_months=None):
    """
    Prioritize images for a tile based on temporal distance.
    
    Parameters:
    -----------
    footprints_subset : geopandas.GeoDataFrame
        Footprints that intersect with the tile
    target_year : int
        Target year for mosaic
    target_doy : int, default=212
        Target day of year (212 = July 31)
    delta_years : int, default=5
        Maximum year difference to include (±delta_years from target)
    include_months : list of int, optional
        List of months to include (e.g., [6, 7, 8] for June-August).
        If None, all months are included.
    
    Returns:
    --------
    geopandas.GeoDataFrame
        Sorted footprints with temporal_score column
    """
    # Create target date
    target_date = datetime(target_year, 1, 1)
    
    # Filter by year range
    year_min = target_year - delta_years
    year_max = target_year + delta_years
    
    subset = footprints_subset[
        (footprints_subset['year'] >= year_min) & 
        (footprints_subset['year'] <= year_max)
    ].copy()
    
    # Filter by months if specified
    if include_months is not None:
        if len(subset) > 0:
            n_before_month_filter = len(subset)
            subset = subset[subset['month'].isin(include_months)]
            n_after_month_filter = len(subset)
            if n_after_month_filter < n_before_month_filter:
                print(f"    Month filter: {n_before_month_filter} → {n_after_month_filter} images "
                      f"(keeping months: {include_months})")
    
    if len(subset) == 0:
        return subset
    
    # Calculate temporal scores
    subset['temporal_score'] = subset.apply(
        lambda row: calculate_temporal_distance(
            datetime(int(row['year']), int(row['month']), int(row['day'])),
            target_date,
            target_doy
        ),
        axis=1
    )
    
    # Sort by temporal score (closest first)
    subset = subset.sort_values('temporal_score')
    
    return subset

# =============================================================================
# MASK HANDLING
# =============================================================================

def apply_mask_from_coarse_data(target_raster_path, mask_footprints_gdf, 
                                 target_bounds, target_crs, target_height, target_width):
    """
    Apply valid data mask from coarser resolution data to target raster.
    
    Parameters:
    -----------
    target_raster_path : str
        Path to target raster file
    mask_footprints_gdf : geopandas.GeoDataFrame
        Footprints of mask files with 'path' column
    target_bounds : tuple
        Bounds of target tile (minx, miny, maxx, maxy)
    target_crs : CRS
        CRS of target tile
    
    Returns:
    --------
    numpy.ndarray or None
        Updated mask array (True = valid, False = invalid)
    """
    # Find mask files that intersect with target bounds
    target_box = box(*target_bounds)
    target_gdf = gpd.GeoDataFrame({'geometry': [target_box]}, crs=target_crs)
    
    # Reproject if needed
    if mask_footprints_gdf.crs != target_crs:
        mask_footprints_projected = mask_footprints_gdf.to_crs(target_crs)
    else:
        mask_footprints_projected = mask_footprints_gdf
    
    # Find intersecting mask files
    intersecting_masks = mask_footprints_projected[
        mask_footprints_projected.intersects(target_box)
    ]
    
    if len(intersecting_masks) == 0:
        return None
    
    # Use the provided target dimensions (from the VRT)
    # Calculate transform for the target tile
    target_transform = rasterio.transform.from_bounds(
        *target_bounds,
        target_width,
        target_height
    )
    
    # Initialize combined mask
    combined_mask = np.zeros((target_height, target_width), dtype=bool)
    
    # Process each mask file
    for idx, mask_row in intersecting_masks.iterrows():
        # Construct full path from 'path' and 'file' fields
        mask_path = os.path.join(mask_row['path'], mask_row['file'])
        
        if not os.path.exists(mask_path):
            print(f"  Warning: Mask file not found: {mask_path}")
            continue
        
        try:
            with rasterio.open(mask_path) as mask_src:
                # Read mask data
                mask_data = mask_src.read(1)
                
                # Reproject mask to target resolution and extent
                mask_reprojected = np.zeros((target_height, target_width), 
                                             dtype=mask_data.dtype)
                
                reproject(
                    source=mask_data,
                    destination=mask_reprojected,
                    src_transform=mask_src.transform,
                    src_crs=mask_src.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
                
                # Update combined mask (True = valid)
                # Assume mask values: 0 = invalid, >0 = valid
                combined_mask = combined_mask | (mask_reprojected > 0)
                
        except Exception as e:
            print(f"  Warning: Failed to process mask {mask_path}: {e}")
            continue
    
    return combined_mask
    
# =============================================================================
# TILE MOSAICKING
# =============================================================================

def mosaic_tile(tile_row, grid_gdf, footprints_gdf, output_dir, target_year, 
                target_doy=212, delta_years=5, output_resolution=0.5,
                mask_footprints_gdf=None, mosaic_method='first',
                include_months=None):
    """
    Create mosaic for a single grid tile.
    
    Parameters:
    -----------
    tile_row : pandas.Series
        Row from grid_gdf containing tile information
    grid_gdf : geopandas.GeoDataFrame
        Full grid GeoDataFrame
    footprints_gdf : geopandas.GeoDataFrame
        Footprints with columns: ['path', 'year', 'month', 'day', 'geometry']
    output_dir : str
        Directory for output files
    target_year : int
        Target year for mosaic
    target_doy : int, default=212
        Target day of year
    delta_years : int, default=5
        Maximum year difference
    output_resolution : float, default=0.5
        Output resolution in units of target CRS (meters for projected)
    mask_footprints_gdf : geopandas.GeoDataFrame, optional
        Footprints for coarse mask data
    mosaic_method : str, default='first'
        Mosaic method: 'first', 'last', 'mean', 'min', 'max'
    
    Returns:
    --------
    dict
        Results dictionary with keys: 'tile_id', 'success', 'output_path', 'message'
    """
    tile_id = tile_row['tile_id']
    tile_geom = tile_row['geometry']
    
    print(f"\n{'='*60}")
    print(f"Processing tile: {tile_id}")
    print(f"{'='*60}")
    
    # Find intersecting footprints
    footprints_in_tile = footprints_gdf[footprints_gdf.intersects(tile_geom)].copy()
    
    if len(footprints_in_tile) == 0:
        print(f"  No images intersect tile {tile_id}, skipping...")
        return {
            'tile_id': tile_id,
            'success': False,
            'output_path': None,
            'message': 'No intersecting images'
        }
    
    print(f"  Found {len(footprints_in_tile)} intersecting images")
    
    # Prioritize images by temporal distance
    prioritized_fps = prioritize_images_for_tile(
        footprints_in_tile, 
        target_year, 
        target_doy, 
        delta_years,
        include_months
    )
    
    if len(prioritized_fps) == 0:
        print(f"  No images within ±{delta_years} years of {target_year}, skipping...")
        return {
            'tile_id': tile_id,
            'success': False,
            'output_path': None,
            'message': f'No images within ±{delta_years} years'
        }
    
    print(f"  {len(prioritized_fps)} images within temporal range")
    print(f"  Temporal scores range: {prioritized_fps['temporal_score'].min():.1f} - "
          f"{prioritized_fps['temporal_score'].max():.1f} days")
    
    # Get tile bounds and CRS
    tile_bounds = tile_geom.bounds  # minx, miny, maxx, maxy
    tile_crs = grid_gdf.crs
    
    # Prepare list of rasters to mosaic
    # Initialize output arrays first
    # Calculate output dimensions
    output_width = int((tile_bounds[2] - tile_bounds[0]) / output_resolution)
    output_height = int((tile_bounds[3] - tile_bounds[1]) / output_resolution)
    output_transform = rasterio.transform.from_bounds(
        *tile_bounds,
        output_width,
        output_height
    )
    
    # Get metadata from first valid raster
    first_raster_meta = None
    n_valid_files = 0
    for idx, fp_row in prioritized_fps.iterrows():
        raster_path = os.path.join(fp_row['path'], fp_row['file'])
        
        if not os.path.exists(raster_path):
            print(f"  Warning: File not found: {raster_path}")
            continue
            
        n_valid_files += 1
        
        try:
            with rasterio.open(raster_path) as src:
                first_raster_meta = {
                    'count': src.count,
                    'dtype': src.dtypes[0],  # Get dtype from first band
                    'nodata': src.nodata if src.nodata is not None else 0
                }
                print(f"  Using metadata from: {os.path.basename(raster_path)}")
                print(f"    Bands: {src.count}, Dtype: {src.dtypes[0]}, Nodata: {src.nodata}")
                break
        except Exception as e:
            print(f"  Warning: Failed to open {raster_path}: {e}")
            continue
    
    if first_raster_meta is None:
        if n_valid_files == 0:
            print(f"  No files exist on disk for tile {tile_id}")
        else:
            print(f"  {n_valid_files} files exist but could not be opened for tile {tile_id}")
        return {
            'tile_id': tile_id,
            'success': False,
            'output_path': None,
            'message': f'No valid rasters (found {n_valid_files} files)'
        }
    
    n_bands = first_raster_meta['count']
    nodata_value = first_raster_meta['nodata']
    
    # Initialize 3-band output array
    # Band 1: Raster value (vegetation height)
    # Band 2: Year
    # Band 3: Day of Year
    mosaic_data = np.full((3, output_height, output_width), 
                          nodata_value, 
                          dtype=first_raster_meta['dtype'])
    
    # Create a mask to track which pixels have been filled
    filled_mask = np.zeros((output_height, output_width), dtype=bool)
    
    # Group rasters by temporal score to handle ties
    score_groups = {}
    for idx, fp_row in prioritized_fps.iterrows():
        score = fp_row['temporal_score']
        if score not in score_groups:
            score_groups[score] = []
        score_groups[score].append(fp_row)
    
    print(f"  Processing {len(score_groups)} temporal score groups")
    
    # Process rasters in order of temporal score - ONE AT A TIME
    for score in sorted(score_groups.keys()):
        group = score_groups[score]
        
        if len(group) == 1:
            # Single raster for this temporal score - use 'first' method
            fp_row = group[0]
            raster_path = os.path.join(fp_row['path'], fp_row['file'])
            
            if not os.path.exists(raster_path):
                continue
            
            try:
                # Open and warp raster - process immediately, don't store
                with rasterio.open(raster_path) as src:
                    with WarpedVRT(
                        src,
                        crs=tile_crs,
                        resampling=Resampling.bilinear,
                        transform=output_transform,
                        width=output_width,
                        height=output_height
                    ) as vrt:
                        # Read data
                        data = vrt.read()
                        
                        # Apply mask from coarse data if provided
                        if mask_footprints_gdf is not None:
                            mask = apply_mask_from_coarse_data(
                                raster_path, 
                                mask_footprints_gdf,
                                tile_bounds,
                                tile_crs,
                                output_height,
                                output_width
                            )
                            if mask is not None and mask.shape == (output_height, output_width):
                                # Apply mask to all bands
                                for band in range(n_bands):
                                    data[band][~mask] = nodata_value
                        
                        # Find valid pixels that haven't been filled yet
                        valid_mask = (data[0] != nodata_value) & ~filled_mask
                        
                        # Calculate DOY
                        doy_val = datetime(int(fp_row['year']), 
                                         int(fp_row['month']), 
                                         int(fp_row['day'])).timetuple().tm_yday
                        
                        # Fill mosaic - Band 1: raster value, Band 2: year, Band 3: DOY
                        mosaic_data[0][valid_mask] = data[0][valid_mask]  # Raster value
                        mosaic_data[1][valid_mask] = fp_row['year']       # Year
                        mosaic_data[2][valid_mask] = doy_val              # Day of Year
                        
                        filled_mask = filled_mask | valid_mask
                        
                        # Free memory
                        del data
                        
            except Exception as e:
                print(f"  Warning: Failed to read {raster_path}: {e}")
                continue
            
        else:
            # Multiple rasters with same temporal score - use 'mean'
            print(f"  Using mean for {len(group)} rasters with score {score}")
            
            # Initialize accumulators for the raster value (band 1 only)
            sum_data = np.zeros((output_height, output_width), dtype=np.float64)
            count_data = np.zeros((output_height, output_width), dtype=np.int16)
            
            # Read each raster one at a time
            for fp_row in group:
                raster_path = os.path.join(fp_row['path'], fp_row['file'])
                
                if not os.path.exists(raster_path):
                    continue
                
                try:
                    with rasterio.open(raster_path) as src:
                        with WarpedVRT(
                            src,
                            crs=tile_crs,
                            resampling=Resampling.bilinear,
                            transform=output_transform,
                            width=output_width,
                            height=output_height
                        ) as vrt:
                            # Read only the first band (raster value)
                            band_data = vrt.read(1)
                            
                            # Apply mask if provided
                            if mask_footprints_gdf is not None:
                                mask = apply_mask_from_coarse_data(
                                    raster_path,
                                    mask_footprints_gdf,
                                    tile_bounds,
                                    tile_crs,
                                    output_height,
                                    output_width
                                )
                                if mask is not None and mask.shape == (output_height, output_width):
                                    band_data[~mask] = nodata_value
                            
                            # Find valid pixels
                            valid = (band_data != nodata_value) & ~filled_mask
                            
                            # Accumulate
                            sum_data[valid] += band_data[valid]
                            count_data[valid] += 1
                            
                            # Free memory
                            del band_data
                            
                except Exception as e:
                    print(f"  Warning: Failed to read {raster_path}: {e}")
                    continue
            
            # Calculate mean for raster values
            valid_mean = (count_data > 0) & ~filled_mask
            if np.any(valid_mean):
                mean_data = sum_data / np.maximum(count_data, 1)
                mosaic_data[0][valid_mean] = mean_data[valid_mean].astype(mosaic_data.dtype)
                
                # For year/DOY when there are ties, use first raster's dates
                fp_row = group[0]
                doy_val = datetime(int(fp_row['year']), 
                                 int(fp_row['month']), 
                                 int(fp_row['day'])).timetuple().tm_yday
                
                mosaic_data[1][valid_mean] = fp_row['year']  # Year
                mosaic_data[2][valid_mean] = doy_val         # DOY
                filled_mask = filled_mask | valid_mean
            
            # Free memory
            del sum_data, count_data
    
    # Calculate statistics
    percent_filled = (np.sum(filled_mask) / (output_height * output_width)) * 100
    print(f"  Mosaic coverage: {percent_filled:.1f}%")
    
    if percent_filled < 1:
        print(f"  Warning: Very low coverage for tile {tile_id}")
    
    # Define output path (single 3-band file)
    mosaic_filename = f"mosaic_{target_year}_DOY{target_doy:03d}_{tile_id}.tif"
    mosaic_path = os.path.join(output_dir, mosaic_filename)
    
    # Write 3-band mosaic (as COG)
    try:
        profile = {
            'driver': 'GTiff',
            'dtype': mosaic_data.dtype,
            'width': output_width,
            'height': output_height,
            'count': 3,  # 3 bands: value, year, DOY
            'crs': tile_crs,
            'transform': output_transform,
            'nodata': nodata_value,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
            'BIGTIFF': 'IF_SAFER'
        }
        
        # Write mosaic
        with rasterio.open(mosaic_path, 'w', **profile) as dst:
            dst.write(mosaic_data)
            # Set band descriptions
            dst.set_band_description(1, 'Raster_Value')
            dst.set_band_description(2, 'Year')
            dst.set_band_description(3, 'Day_of_Year')
        
        # Convert to COG using gdal_translate
        mosaic_path_cog = mosaic_path.replace('.tif', '_COG.tif')
        gdal.Translate(
            mosaic_path_cog,
            mosaic_path,
            format='COG',
            creationOptions=['COMPRESS=LZW', 'BIGTIFF=IF_SAFER']
        )
        
        # Remove non-COG version
        os.remove(mosaic_path)
        
        # Rename COG to original name
        os.rename(mosaic_path_cog, mosaic_path)
        
        print(f"  3-band mosaic written: {mosaic_path}")
        print(f"    Band 1: Raster Value")
        print(f"    Band 2: Year")
        print(f"    Band 3: Day of Year")
        
        return {
            'tile_id': tile_id,
            'success': True,
            'output_path': mosaic_path,
            'coverage_percent': percent_filled,
            'n_images_used': len(score_groups),  # Number of temporal score groups
            'message': 'Success'
        }
        
    except Exception as e:
        print(f"  Error writing outputs for tile {tile_id}: {e}")
        return {
            'tile_id': tile_id,
            'success': False,
            'output_path': None,
            'message': f'Write error: {e}'
        }

# =============================================================================
# VRT CREATION
# =============================================================================

def create_vrt_from_tiles(mosaic_tiles_dir, output_vrt_path, pattern='mosaic_*.tif'):
    """
    Create VRT from mosaic tiles.
    
    Parameters:
    -----------
    mosaic_tiles_dir : str
        Directory containing mosaic tiles
    output_vrt_path : str
        Output VRT file path
    pattern : str, default='mosaic_*.tif'
        Glob pattern to match tile files
    
    Returns:
    --------
    str
        Path to created VRT
    """
    from pathlib import Path
    import glob
    
    # Find all mosaic tiles
    tile_files = sorted(glob.glob(os.path.join(mosaic_tiles_dir, pattern)))
    
    if len(tile_files) == 0:
        print(f"No tiles found matching pattern: {pattern}")
        return None
    
    print(f"\nCreating VRT from {len(tile_files)} tiles...")
    
    # Use gdalbuildvrt to create VRT
    try:
        # Build VRT command
        cmd = ['gdalbuildvrt', output_vrt_path] + tile_files
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"VRT created: {output_vrt_path}")
            return output_vrt_path
        else:
            print(f"Error creating VRT: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Error creating VRT: {e}")
        return None

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def create_gridded_mosaic(footprints_gpkg, aoi_geom, output_dir, target_year,
                         target_doy=212, delta_years=5, grid_size_km=90,
                         output_resolution=0.5, target_crs=None,
                         mask_footprints_gpkg=None, n_jobs=-1,
                         include_months=None):
    """
    Create gridded mosaic with temporal prioritization.
    
    Parameters:
    -----------
    footprints_gpkg : str
        Path to footprints geopackage with columns:
        ['path', 'year', 'month', 'day', 'geometry']
    aoi_geom : various types
        Area of interest. Can be:
        - String 'alaska' for automatic Alaska extent
        - Shapely geometry
        - GeoDataFrame
        - Path to vector file (shapefile, geopackage, etc.)
    output_dir : str
        Directory for output files
    target_year : int
        Target year for mosaic
    target_doy : int, default=212
        Target day of year (212 = July 31)
    delta_years : int, default=5
        Maximum year difference (±delta_years)
    grid_size_km : float, default=90
        Grid cell size in kilometers
    output_resolution : float, default=0.5
        Output resolution in target CRS units (meters for projected)
    target_crs : str, optional
        Target CRS. If None, automatically determined
    mask_footprints_gpkg : str, optional
        Path to coarse mask footprints geopackage
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all cores)
    include_months : list of int, optional
        List of months to include (e.g., [6, 7, 8] for summer months).
        If None, all months are included. Default is None.
    Returns:
    --------
    tuple
        (results_df, vrt_path)
        - results_df: DataFrame with processing results for each tile
        - vrt_path: Path to mosaic VRT file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("GRIDDED MOSAIC CREATION")
    print("="*80)
    print(f"Target year: {target_year}")
    print(f"Target DOY: {target_doy} (±{delta_years} years)")
    if include_months is not None:
        print(f"Include months: {include_months}")
    else:
        print(f"Include months: All")
    print(f"Grid size: {grid_size_km} km")
    print(f"Output resolution: {output_resolution} m")
    print(f"Output directory: {output_dir}")
    
    # Load footprints
    print("\nLoading footprints...")
    footprints_gdf = gpd.read_file(footprints_gpkg)
    print(f"Loaded {len(footprints_gdf)} footprints")
    
    # Parse date field into year, month, day if not already present
    if 'date' in footprints_gdf.columns:
        print("Parsing 'date' field into year, month, day...")
        # Convert to datetime if it's not already
        footprints_gdf['date'] = pd.to_datetime(footprints_gdf['date'])
        footprints_gdf['year'] = footprints_gdf['date'].dt.year
        footprints_gdf['month'] = footprints_gdf['date'].dt.month
        footprints_gdf['day'] = footprints_gdf['date'].dt.day
        print(f"  Date range: {footprints_gdf['date'].min()} to {footprints_gdf['date'].max()}")
    
    # Verify required columns
    required_cols = ['path', 'year', 'month', 'day', 'geometry']
    missing_cols = [col for col in required_cols if col not in footprints_gdf.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in footprints: {missing_cols}")
    
    # Load mask footprints if provided
    mask_footprints_gdf = None
    if mask_footprints_gpkg is not None:
        print("\nLoading mask footprints...")
        mask_footprints_gdf = gpd.read_file(mask_footprints_gpkg)
        print(f"Loaded {len(mask_footprints_gdf)} mask footprints")
    
    # Load or process AOI
    if isinstance(aoi_geom, str):
        if aoi_geom.lower() == 'alaska':
            aoi_for_grid = 'alaska'
        elif os.path.exists(aoi_geom):
            # Load from file
            aoi_for_grid = gpd.read_file(aoi_geom)
        else:
            raise ValueError(f"AOI string not recognized: {aoi_geom}")
    else:
        aoi_for_grid = aoi_geom
    
    # Create grid
    print("\nCreating vector grid...")
    grid_gdf = create_vector_grid(aoi_for_grid, grid_size_km, target_crs)
    
    # Save grid
    grid_path = os.path.join(output_dir, 'mosaic_grid.gpkg')
    grid_gdf.to_file(grid_path, driver='GPKG')
    print(f"Grid saved: {grid_path}")
    
    # Reproject footprints to grid CRS if needed
    if footprints_gdf.crs != grid_gdf.crs:
        print("\nReprojecting footprints to grid CRS...")
        footprints_gdf = footprints_gdf.to_crs(grid_gdf.crs)
    
    if mask_footprints_gdf is not None and mask_footprints_gdf.crs != grid_gdf.crs:
        print("Reprojecting mask footprints to grid CRS...")
        mask_footprints_gdf = mask_footprints_gdf.to_crs(grid_gdf.crs)
    
    # Process tiles in parallel
    print("\n" + "="*80)
    print(f"PROCESSING {len(grid_gdf)} TILES (parallel jobs: {n_jobs})")
    print("="*80)
    
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(mosaic_tile)(
            row, grid_gdf, footprints_gdf, output_dir, target_year,
            target_doy, delta_years, output_resolution, mask_footprints_gdf,
            'first', include_months
        )
        for idx, row in grid_gdf.iterrows()
    )
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join(output_dir, 'processing_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved: {results_path}")
    
    # Print summary
    n_success = results_df['success'].sum()
    n_failed = len(results_df) - n_success
    
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Total tiles: {len(results_df)}")
    print(f"Successful: {n_success}")
    print(f"Failed: {n_failed}")
    
    if n_success > 0:
        avg_coverage = results_df[results_df['success']]['coverage_percent'].mean()
        print(f"Average coverage: {avg_coverage:.1f}%")
    
    # Create VRTs
    print("\n" + "="*80)
    print("CREATING VRTs")
    print("="*80)
    
    mosaic_vrt_path = os.path.join(output_dir, 
                                   f'mosaic_{target_year}_DOY{target_doy:03d}.vrt')
    
    create_vrt_from_tiles(output_dir, mosaic_vrt_path, 
                         pattern=f'mosaic_{target_year}_DOY{target_doy:03d}_*.tif')
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    
    return results_df, mosaic_vrt_path
