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

# Flag values:
#  0  = Valid data
# -1  = Non-valid data (clouds/cloud shadows/quality issues from dm-10m mask)
# -2  = No images available (no coverage in time/month window)
# -3  = Outside AOI boundary

Author: Paul Montesano
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
            print(f"    Warning: Mask file not found: {mask_path}")
            continue
        
        # Print mask date info if available
        mask_date_info = ""
        if 'date' in mask_row:
            mask_date_info = f" | Date: {mask_row['date']}"
        if 'month' in mask_row:
            mask_date_info += f" | Month: {mask_row['month']}"
        
        try:
            with rasterio.open(mask_path) as mask_src:
                print(f"    Processing mask: {os.path.basename(mask_path)}{mask_date_info}")
                print(f"      Mask CRS: {mask_src.crs}")
                print(f"      Target CRS: {target_crs}")
                print(f"      Mask shape: {mask_src.shape}")
                print(f"      Target shape: ({target_height}, {target_width})")
                
                # Read mask data
                mask_data = mask_src.read(1)
                mask_nodata = mask_src.nodata
                
                #print(f"      Mask nodata value: {mask_nodata}")
                #print(f"      Mask unique values: {np.unique(mask_data)}")
                
                # Initialize reprojected mask
                mask_reprojected = np.zeros((target_height, target_width), 
                                             dtype=mask_data.dtype)
                
                # Reproject mask from Albers to UTM (or whatever target CRS is)
                reproject(
                    source=mask_data,
                    destination=mask_reprojected,
                    src_transform=mask_src.transform,
                    src_crs=mask_src.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest  # Use nearest for categorical mask data
                )
                
                #print(f"      Reprojected unique values: {np.unique(mask_reprojected)}")
                
                # Determine valid pixels based on mask values
                # For dm-10m masks: valid data is >= 0, invalid is nodata
                if mask_nodata is not None:
                    # Valid = anything that is NOT nodata (including 0)
                    valid_in_mask = (mask_reprojected != mask_nodata)
                    print(f"      Using nodata={mask_nodata} as invalid, all other values (>=0) as valid")
                    print(f"      Valid value range in mask: {mask_reprojected[valid_in_mask].min():.1f} to {mask_reprojected[valid_in_mask].max():.1f}")
                else:
                    # If no nodata defined, assume all finite values are valid
                    valid_in_mask = np.isfinite(mask_reprojected)
                    print(f"      No nodata value defined, using all finite values as valid")
                
                n_valid = np.sum(valid_in_mask)
                n_total = valid_in_mask.size
                print(f"      Valid pixels after reprojection: {n_valid:,}/{n_total:,} ({100*n_valid/n_total:.1f}%)")
                
                # Update combined mask (True = valid)
                combined_mask = combined_mask | valid_in_mask
                
        except Exception as e:
            print(f"    Warning: Failed to process mask {mask_path}: {e}")
            continue
    
    # Calculate final combined mask statistics
    n_valid_final = np.sum(combined_mask)
    n_total_final = combined_mask.size
    print(f"  Final combined mask: {n_valid_final:,}/{n_total_final:,} valid ({100*n_valid_final/n_total_final:.1f}%)")
    
    return combined_mask

def get_mask_for_raster(raster_path, mask_footprints_gdf, tile_bounds, tile_crs, 
                        target_height, target_width):
    """
    Get the specific mask for a single raster file.
    
    Matches by finding the dm-10m file that corresponds to the 2m CHM file.
    """
    import os
    
    # Extract base name from raster (remove -sr-02m.chm.tif, add -chm-dm-10m.tif)
    raster_basename = os.path.basename(raster_path)
    
    # Convert CHM filename to DM filename
    # Example: WV02_20180831_M1BS_1030010081AF0800-sr-02m.chm.tif
    #       -> WV02_20180831_M1BS_1030010081AF0800-chm-dm-10m.tif
    dm_filename = raster_basename.replace('-sr-02m.chm.tif', '-chm-dm-10m.tif')
    
    # Find the matching mask in mask_footprints_gdf
    matching_masks = mask_footprints_gdf[mask_footprints_gdf['file'] == dm_filename]
    
    if len(matching_masks) == 0:
        print(f"    Warning: No matching dm-10m mask found for {raster_basename}")
        return None
    
    if len(matching_masks) > 1:
        print(f"    Warning: Multiple masks found for {raster_basename}, using first")
    
    # Get the mask file path
    mask_row = matching_masks.iloc[0]
    mask_path = os.path.join(mask_row['path'], mask_row['file'])
    
    if not os.path.exists(mask_path):
        print(f"    Warning: Mask file not found: {mask_path}")
        return None
    
    print(f"    Loading mask: {dm_filename}")
    
    # Load and reproject the mask
    try:
        with rasterio.open(mask_path) as mask_src:
            mask_data = mask_src.read(1)
            mask_nodata = mask_src.nodata
            
            # Initialize reprojected mask
            mask_reprojected = np.zeros((target_height, target_width), dtype=mask_data.dtype)
            
            # Reproject mask
            reproject(
                source=mask_data,
                destination=mask_reprojected,
                src_transform=mask_src.transform,
                src_crs=mask_src.crs,
                dst_transform=rasterio.transform.from_bounds(*tile_bounds, target_width, target_height),
                dst_crs=tile_crs,
                resampling=Resampling.nearest
            )
            
            # Determine valid pixels
            if mask_nodata is not None:
                valid_mask = (mask_reprojected != mask_nodata)
            else:
                valid_mask = np.isfinite(mask_reprojected)
            
            n_valid = np.sum(valid_mask)
            print(f"      Valid pixels: {n_valid:,}/{valid_mask.size:,} ({100*n_valid/valid_mask.size:.1f}%)")
            
            return valid_mask
            
    except Exception as e:
        print(f"    Warning: Failed to load mask {mask_path}: {e}")
        return None

# =============================================================================
# TILE MOSAICKING
# =============================================================================

def mosaic_tile(tile_row, grid_gdf, footprints_gdf, output_dir, target_year, 
                target_doy=212, delta_years=5, output_resolution=0.5,
                mask_footprints_gdf=None, mosaic_method='first',
                include_months=None, aoi_mask=None):
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
    
    # Get metadata from first valid raster (for data type and structure only)
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
                    'dtype': src.dtypes[0],
                    'nodata': src.nodata if src.nodata is not None else 0
                }
                print(f"  Reading metadata (dtype, nodata) from: {os.path.basename(raster_path)}")
                print(f"    Dtype: {src.dtypes[0]}, Nodata: {src.nodata}")
                print(f"    Note: Year/DOY will be assigned per-pixel based on source raster")
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
    
    # Initialize 4-band output array
    # Band 1: Raster value (vegetation height)
    # Band 2: Year
    # Band 3: Day of Year
    # Band 4: Nodata flag
    mosaic_data = np.full((3, output_height, output_width), 
                          nodata_value, 
                          dtype=first_raster_meta['dtype'])
    
    # Initialize flag band (int8 to save space)
    # Flags: 0=valid data, -1=non-valid (cloud/shadows/out-of-raster-data-extent/quality), -2=no images, -3=outside AOI
    flag_band = np.zeros((output_height, output_width), dtype=np.int8)
    
    # Create masks to track different conditions
    filled_mask = np.zeros((output_height, output_width), dtype=bool)
    non_valid_pixels = np.zeros((output_height, output_width), dtype=bool)  # Tracks dm masked pixels
    
    # Create list of all rasters with their temporal scores
    raster_list = []
    for idx, fp_row in prioritized_fps.iterrows():
        raster_path = os.path.join(fp_row['path'], fp_row['file'])
        if os.path.exists(raster_path):
            raster_list.append({
                'path': raster_path,
                'year': fp_row['year'],
                'month': fp_row['month'],
                'day': fp_row['day'],
                'temporal_score': fp_row['temporal_score']
            })

    # ADD THIS DIAGNOSTIC
    if raster_list:
        months_in_list = sorted(set(r['month'] for r in raster_list))
        print(f"  Months in raster_list: {months_in_list}")
        if include_months is not None:
            unexpected_months = set(months_in_list) - set(include_months)
            if unexpected_months:
                print(f"  WARNING: Found unexpected months: {unexpected_months}")
    
    if len(raster_list) == 0:
        print(f"  No valid raster files found for tile {tile_id}")
        return {
            'tile_id': tile_id,
            'success': False,
            'output_path': None,
            'message': 'No valid raster files'
        }
    
    print(f"  Processing {len(raster_list)} rasters with priority filling")
    print(f"  Temporal score range: {min(r['temporal_score'] for r in raster_list):.0f} - "
          f"{max(r['temporal_score'] for r in raster_list):.0f} days")
    
    # Initialize arrays to store best value and its priority for each pixel
    # Priority is temporal_score (lower = better)
    best_value = np.full((output_height, output_width), nodata_value, dtype=first_raster_meta['dtype'])
    best_priority = np.full((output_height, output_width), np.inf, dtype=np.float32)
    best_year = np.zeros((output_height, output_width), dtype=np.int16)
    best_doy = np.zeros((output_height, output_width), dtype=np.int16)
    
    # Track which pixels have been filled
    filled_mask = np.zeros((output_height, output_width), dtype=bool)
    
    # Process each raster in order of temporal priority
    n_rasters_used = 0
    for raster_info in raster_list:
        raster_path = raster_info['path']
        temporal_score = raster_info['temporal_score']
        
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
                    # Read data
                    data = vrt.read(1)
                    
                    # Apply raster-specific mask if available
                    if mask_footprints_gdf is not None:
                        raster_mask = get_mask_for_raster(
                            raster_path,
                            mask_footprints_gdf,
                            tile_bounds,
                            tile_crs,
                            output_height,
                            output_width
                        )
                        
                        if raster_mask is not None and raster_mask.shape == (output_height, output_width):
                            n_valid_before = np.sum(data != nodata_value)
                            data[~raster_mask] = nodata_value
                            n_valid_after = np.sum(data != nodata_value)
                            n_masked = n_valid_before - n_valid_after
                            if n_masked > 0:
                                print(f"      Masked out {n_masked:,} pixels ({100*n_masked/n_valid_before:.1f}% of valid data)")
                            
                            # Track non-valid pixels for flag assignment
                            non_valid_pixels = non_valid_pixels | ~raster_mask
                    
                    # Find valid pixels in this raster
                    valid_pixels = (data != nodata_value)
                    
                    # Find pixels where this raster has better (lower) priority than current best
                    # OR where current best is still nodata
                    better_priority = (temporal_score < best_priority) | ((best_value == nodata_value) & valid_pixels)
                    
                    # Update pixels that have better priority
                    update_mask = valid_pixels & better_priority
                    
                    if np.any(update_mask):
                        best_value[update_mask] = data[update_mask]
                        best_priority[update_mask] = temporal_score
                        
                        # Calculate DOY
                        doy_val = datetime(int(raster_info['year']), 
                                         int(raster_info['month']), 
                                         int(raster_info['day'])).timetuple().tm_yday
                        
                        best_year[update_mask] = raster_info['year']
                        best_doy[update_mask] = doy_val
                        filled_mask = filled_mask | update_mask
                        
                        n_updated = np.sum(update_mask)
                        n_rasters_used += 1
                        
                        # Format date string
                        date_str = f"{raster_info['year']}-{raster_info['month']:02d}-{raster_info['day']:02d}"
                        
                        print(f"    Raster {n_rasters_used}/{len(raster_list)}: {os.path.basename(raster_path)}")
                        print(f"      Date: {date_str} | Temporal score: {temporal_score:.0f} days")
                        print(f"      Pixels used: {n_updated:,} | Total valid in raster: {np.sum(valid_pixels):,}")
                    
                    # Free memory
                    del data
                    
        except Exception as e:
            print(f"  Warning: Failed to read {raster_path}: {e}")
            continue
    
    # Assign final values to mosaic bands
    mosaic_data[0] = best_value   # Band 1: Raster value
    mosaic_data[1] = best_year    # Band 2: Year
    mosaic_data[2] = best_doy     # Band 3: Day of Year
    
    # ============================================================================
    # ASSIGN FLAGS IN PRIORITY ORDER (most specific to least specific)
    # ============================================================================
    
    # Start with all pixels = 0 (will be overwritten for nodata pixels)
    flag_band[:] = 0
    
    # 1. Flag non-valid pixels (clouds/shadows/quality) = -1
    #    These pixels are masked by dm-10m (not valid data)
    flag_band[non_valid_pixels] = -1
    
    # 2. Flag pixels with no images available in time window = -2
    #    These pixels had no coverage at all in the specified time/month window
    no_images_available = ~filled_mask & ~non_valid_pixels
    flag_band[no_images_available] = -2
    
    # 3. Valid data = 0
    #    Pixels that were successfully filled keep flag = 0
    flag_band[filled_mask] = 0
    
    print(f"\n  === FLAG ASSIGNMENT (before AOI mask) ===")
    print(f"    Valid data (0):                                         {np.sum(flag_band == 0):,} pixels")
    print(f"    Non-valid data (cloud/shadows/s/outside extent) (-1):   {np.sum(flag_band == -1):,} pixels")
    print(f"    No images available (-2):                               {np.sum(flag_band == -2):,} pixels")
    
    # Apply AOI mask as final filter if provided
    if aoi_mask is not None:
        print(f"\n  Applying AOI mask...")
        try:
            # Rasterize AOI to match output extent
            from rasterio.features import rasterize
            from shapely.geometry import mapping
            
            # Create AOI mask array
            aoi_shapes = [(mapping(geom), 1) for geom in aoi_mask.geometry]
            aoi_raster = rasterize(
                aoi_shapes,
                out_shape=(output_height, output_width),
                transform=output_transform,
                fill=0,
                dtype=np.uint8
            )
            
            # Convert to boolean (1 = inside AOI, 0 = outside)
            aoi_valid = aoi_raster > 0
            
            # Count pixels before masking
            n_valid_before = np.sum(filled_mask)
            
            # Apply AOI mask (set pixels outside AOI to nodata)
            outside_aoi = ~aoi_valid
            
            # Set data to nodata for pixels outside AOI
            mosaic_data[0][outside_aoi] = nodata_value
            mosaic_data[1][outside_aoi] = 0
            mosaic_data[2][outside_aoi] = 0
            
            # 5. Flag pixels outside AOI = -3 (FINAL override)
            #    This is the most definitive mask - these pixels shouldn't be analyzed
            flag_band[outside_aoi] = -3
            
            filled_mask = filled_mask & aoi_valid
            
            print(f"  === FLAG ASSIGNMENT (after AOI mask) ===")
            
            # Count pixels after masking
            n_valid_after = np.sum(filled_mask)
            n_removed = n_valid_before - n_valid_after
            
            if n_removed > 0:
                print(f"    Removed {n_removed:,} pixels outside AOI ({100*n_removed/n_valid_before:.1f}%)")
                print(f"    Remaining: {n_valid_after:,} pixels inside AOI")
            else:
                print(f"    All pixels are within AOI")
                
        except Exception as e:
            print(f"    Warning: Failed to apply AOI mask: {e}")
    
    # Print summary of raster contributions
    print(f"\n  === RASTER USAGE SUMMARY ===")
    print(f"  Total rasters considered: {len(raster_list)}")
    print(f"  Rasters that contributed pixels: {n_rasters_used}")

    # Print flag statistics
    print(f"\n  Nodata flags:")
    print(f"    Valid data (0):                                     {np.sum(flag_band == 0):,} pixels ({100*np.sum(flag_band == 0)/flag_band.size:.1f}%)")
    print(f"    Non-valid (cloud/shadows/outside extent) (-1):      {np.sum(flag_band == -1):,} pixels ({100*np.sum(flag_band == -1)/flag_band.size:.1f}%)")
    print(f"    No images available (-2):                           {np.sum(flag_band == -2):,} pixels ({100*np.sum(flag_band == -2)/flag_band.size:.1f}%)")
    print(f"    Outside AOI (-3):                                   {np.sum(flag_band == -3):,} pixels ({100*np.sum(flag_band == -3)/flag_band.size:.1f}%)")
    
    # Count pixels by year (only among filled pixels)
    year_counts = {}
    unique_years = np.unique(best_year[filled_mask])
    for year in unique_years:
        if year > 0:  # Skip zero values
            count = np.sum((best_year == year) & filled_mask)  # ← Count only filled pixels with this year
            year_counts[year] = count
    
    if year_counts:
        total_filled = np.sum(filled_mask)
        print(f"  Pixels by year:")
        for year in sorted(year_counts.keys()):
            pct = 100 * year_counts[year] / total_filled if total_filled > 0 else 0
            print(f"    {year}: {year_counts[year]:,} pixels ({pct:.1f}%)")
    
    # Add verification that year/DOY are pixel-specific
    if np.sum(filled_mask) > 0:
        unique_year_doy_combos = set()
        for year in np.unique(best_year[filled_mask]):
            if year > 0:
                doys_for_year = np.unique(best_doy[(best_year == year) & filled_mask])
                for doy in doys_for_year:
                    if doy > 0:
                        unique_year_doy_combos.add((int(year), int(doy)))
        
        print(f"\n  Unique year-DOY combinations: {len(unique_year_doy_combos)}")
        if len(unique_year_doy_combos) <= 10:
            for year, doy in sorted(unique_year_doy_combos):
                date_obj = datetime(year, 1, 1) + timedelta(days=doy-1)
                n_pixels = np.sum((best_year == year) & (best_doy == doy) & filled_mask)
                print(f"    {year}-{doy:03d} ({date_obj.strftime('%Y-%m-%d')}): {n_pixels:,} pixels")
    
    # Calculate statistics
    
    # Calculate statistics
    percent_filled = (np.sum(filled_mask) / (output_height * output_width)) * 100
    print(f"  Mosaic coverage: {percent_filled:.1f}%")
    
    if percent_filled < 1:
        print(f"  Warning: Very low coverage for tile {tile_id}")
    
    # Define output path (single 4-band file with delta_years and months in name)
    if include_months is not None:
        months_str = 'months' + ''.join(str(m) for m in sorted(include_months))
    else:
        months_str = 'monthsAll'
    
    mosaic_filename = f"mosaic_{target_year}_DOY{target_doy:03d}_deltayr{delta_years:02d}_{months_str}_{tile_id}.tif"
    mosaic_path = os.path.join(output_dir, mosaic_filename)
    
    # Write 4-band mosaic (as COG)
    try:
        # Stack all bands including flag band
        output_data = np.vstack([mosaic_data, flag_band[np.newaxis, :, :]])
        
        profile = {
            'driver': 'GTiff',
            'dtype': mosaic_data.dtype,
            'width': output_width,
            'height': output_height,
            'count': 4,  # 4 bands: value, year, DOY, flag
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
            dst.write(output_data)
            # Set band descriptions
            dst.set_band_description(1, 'Raster_Value')
            dst.set_band_description(2, 'Year')
            dst.set_band_description(3, 'Day_of_Year')
            dst.set_band_description(4, 'Nodata_Flag')
        
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
        
        print(f"  4-band mosaic written: {mosaic_path}")
        print(f"    Band 1: Raster Value")
        print(f"    Band 2: Year")
        print(f"    Band 3: Day of Year")
        print(f"    Band 4: Nodata Flag (0=valid, -1=non-valid/cloud/shadowss/outside extent, -2=no images, -3=outside AOI)")
        
        return {
            'tile_id': tile_id,
            'success': True,
            'output_path': mosaic_path,
            'coverage_percent': percent_filled,
            'n_images_used': n_rasters_used,  # ← USE n_rasters_used instead
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
                         include_months=None, tile_ids=None):
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
    tile_ids : list of str, optional
        List of specific tile IDs to process (e.g., ['R0001C0002', 'R0003C0005']).
        If None, all tiles in the grid are processed. Default is None.
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
    print(f"Year range: {target_year - delta_years} to {target_year + delta_years} (±{delta_years} years)")
    print(f"Target DOY: {target_doy}")
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

    # Filter footprints by temporal window (years and months) BEFORE spatial operations
    if 'year' in footprints_gdf.columns:
        year_min = target_year - delta_years
        year_max = target_year + delta_years
        
        print(f"\nFiltering footprints by temporal window:")
        print(f"  Year range: {year_min} to {year_max}")
        n_before_year = len(footprints_gdf)
        
        footprints_gdf = footprints_gdf[
            (footprints_gdf['year'] >= year_min) & 
            (footprints_gdf['year'] <= year_max)
        ]
        
        n_after_year = len(footprints_gdf)
        print(f"  After year filter: {n_before_year} → {n_after_year} images")
        
        # Filter by months if specified
        if include_months is not None and 'month' in footprints_gdf.columns:
            n_before_month = len(footprints_gdf)
            footprints_gdf = footprints_gdf[footprints_gdf['month'].isin(include_months)]
            n_after_month = len(footprints_gdf)
            print(f"  After month filter: {n_after_month} images (months: {include_months})")
    
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

    # Parse mask dates if needed
    if 'date' in mask_footprints_gdf.columns and 'month' not in mask_footprints_gdf.columns:
        print("  Parsing mask footprint dates...")
        mask_footprints_gdf['date'] = pd.to_datetime(mask_footprints_gdf['date'])
        mask_footprints_gdf['month'] = mask_footprints_gdf['date'].dt.month
        mask_footprints_gdf['day'] = mask_footprints_gdf['date'].dt.day
        print(f"    Date range: {mask_footprints_gdf['date'].min()} to {mask_footprints_gdf['date'].max()}")
    
    # Check if month column exists
    if 'month' not in mask_footprints_gdf.columns:
        print("  WARNING: 'month' column not found in mask_footprints_gdf!")
        print(f"  Available columns: {mask_footprints_gdf.columns.tolist()}")
    
    # Filter mask footprints by temporal window (years and months)
    if 'year' in mask_footprints_gdf.columns:
        year_min = target_year - delta_years
        year_max = target_year + delta_years
        
        print(f"\nFiltering mask footprints by temporal window:")
        print(f"  Year range: {year_min} to {year_max}")
        print(f"  Before filter: {len(mask_footprints_gdf)} mask images")
        print(f"  Years in mask data: {sorted(mask_footprints_gdf['year'].unique())}")
        
        # Filter by year range
        mask_footprints_gdf = mask_footprints_gdf[
            (mask_footprints_gdf['year'] >= year_min) & 
            (mask_footprints_gdf['year'] <= year_max)
        ]
        
        print(f"  After year filter: {len(mask_footprints_gdf)} mask images")
        print(f"  Remaining years: {sorted(mask_footprints_gdf['year'].unique())}")
        
        # Filter by months if specified
        if include_months is not None and 'month' in mask_footprints_gdf.columns:
            print(f"  Month range: {include_months}")
            print(f"  Before month filter: {len(mask_footprints_gdf)} mask images")
            
            mask_footprints_gdf = mask_footprints_gdf[
                mask_footprints_gdf['month'].isin(include_months)
            ]
            
            print(f"  After month filter: {len(mask_footprints_gdf)} mask images")
            if len(mask_footprints_gdf) > 0:
                print(f"  Remaining months: {sorted(mask_footprints_gdf['month'].unique())}")
            else:
                print(f"  WARNING: No mask images remain after filtering!")
    else:
        print(f"  WARNING: 'year' column not found in mask_footprints_gdf!")
        print(f"  Available columns: {mask_footprints_gdf.columns.tolist()}")
    
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

    # Create AOI mask GeoDataFrame for rasterization
    if isinstance(aoi_for_grid, str) and aoi_for_grid.lower() == 'alaska':
        # For Alaska, create the bbox geometry
        aoi_mask_gdf = gpd.GeoDataFrame({'geometry': [box(-170, 51, -130, 72)]}, crs='EPSG:4326')
    elif isinstance(aoi_for_grid, gpd.GeoDataFrame):
        aoi_mask_gdf = aoi_for_grid.copy()
    else:
        aoi_mask_gdf = aoi_for_grid
    
    # Create grid
    print("\nCreating vector grid...")
    grid_gdf = create_vector_grid(aoi_for_grid, grid_size_km, target_crs)
    
    # Save grid
    grid_path = os.path.join(output_dir, 'mosaic_grid.gpkg')
    grid_gdf.to_file(grid_path, driver='GPKG')
    print(f"Grid saved: {grid_path}")

    # Reproject AOI mask to grid CRS
    if aoi_mask_gdf.crs != grid_gdf.crs:
        print("\nReprojecting AOI to grid CRS...")
        aoi_mask_gdf = aoi_mask_gdf.to_crs(grid_gdf.crs)

    # Filter grid by tile_ids if specified
    if tile_ids is not None:
        n_tiles_before = len(grid_gdf)
        grid_gdf = grid_gdf[grid_gdf['tile_id'].isin(tile_ids)].copy()
        n_tiles_after = len(grid_gdf)
        
        if n_tiles_after == 0:
            raise ValueError(f"No tiles found matching the specified tile_ids: {tile_ids}")
        
        print(f"\nFiltered grid to {n_tiles_after} tiles (from {n_tiles_before} total)")
        print(f"Processing tiles: {sorted(grid_gdf['tile_id'].tolist())}")
    else:
        print(f"\nProcessing all {len(grid_gdf)} tiles")
    
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
            'first', include_months, aoi_mask_gdf
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
    
    # Create months string for filename
    if include_months is not None:
        months_str = 'months' + ''.join(str(m) for m in sorted(include_months))
    else:
        months_str = 'monthsAll'
    
    mosaic_vrt_path = os.path.join(output_dir, 
                                   f'mosaic_{target_year}_DOY{target_doy:03d}_deltayr{delta_years:02d}_{months_str}.vrt')
    
    create_vrt_from_tiles(output_dir, mosaic_vrt_path, 
                         pattern=f'mosaic_{target_year}_DOY{target_doy:03d}_deltayr{delta_years:02d}_{months_str}_*.tif')
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)

# =============================================================================
    # CREATE MOSAIC-LEVEL SUMMARY
    # =============================================================================
    
    print("\n" + "="*80)
    print("CREATING MOSAIC SUMMARY")
    print("="*80)
    
    # Filter successful tiles for summary statistics
    successful_results = results_df[results_df['success'] == True].copy()
    
    if len(successful_results) > 0:
        # Calculate AOI area and coverage area
        # Get AOI area in square kilometers
        aoi_area_km2 = aoi_mask_gdf.geometry.area.sum() / 1_000_000  # Convert m² to km²
        
        # Calculate coverage area from successful tiles
        # Each pixel is output_resolution x output_resolution meters
        pixel_area_m2 = output_resolution * output_resolution
        
        # Sum coverage_percent across all successful tiles weighted by tile area
        # Each tile has the same number of pixels (grid_size_km determines tile size)
        # Calculate pixels per tile
        pixels_per_tile_side = int((grid_size_km * 1000) / output_resolution)
        pixels_per_tile = pixels_per_tile_side * pixels_per_tile_side
        
        # Calculate total valid pixels across all successful tiles
        total_valid_pixels = 0
        for idx, row in successful_results.iterrows():
            n_valid_in_tile = int((row['coverage_percent'] / 100.0) * pixels_per_tile)
            total_valid_pixels += n_valid_in_tile
        
        # Calculate coverage area in km²
        coverage_area_km2 = (total_valid_pixels * pixel_area_m2) / 1_000_000
        
        # Calculate proportion of AOI covered
        aoi_coverage_proportion = coverage_area_km2 / aoi_area_km2 if aoi_area_km2 > 0 else 0
        
        # Extract CHM paths from footprints used in the mosaic
        year_min = target_year - delta_years
        year_max = target_year + delta_years
        
        # Filter footprints to temporal window
        chms_in_window = footprints_gdf[
            (footprints_gdf['year'] >= year_min) & 
            (footprints_gdf['year'] <= year_max)
        ].copy()
        
        # Apply month filter if specified
        if include_months is not None:
            chms_in_window = chms_in_window[chms_in_window['month'].isin(include_months)]
        
        # Create detailed CHM list with temporal information
        chm_list = []
        for idx, row in chms_in_window.iterrows():
            # Calculate DOY
            doy = datetime(int(row['year']), int(row['month']), int(row['day'])).timetuple().tm_yday
            # Calculate temporal score
            temporal_score = abs(doy - target_doy) + (abs(row['year'] - target_year) * 365)
            
            # Extract filename from path
            chm_filename = os.path.basename(row['path'])
            
            chm_list.append({
                'chm_filename': chm_filename,
                'chm_path': row['path'],
                'year': int(row['year']),
                'month': int(row['month']),
                'day': int(row['day']),
                'doy': doy,
                'temporal_score': temporal_score,
                'date': f"{int(row['year'])}-{int(row['month']):02d}-{int(row['day']):02d}"
            })
        
        chm_df = pd.DataFrame(chm_list)
        chm_df = chm_df.sort_values(['temporal_score', 'chm_path']).reset_index(drop=True)
        
        # Save detailed CHM list
        chm_list_path = os.path.join(output_dir, 
                                     f'chm_list_{target_year}_DOY{target_doy:03d}_deltayr{delta_years:02d}_{months_str}.csv')
        chm_df.to_csv(chm_list_path, index=False)
        print(f"CHM list saved: {chm_list_path}")
        print(f"  Total CHMs in temporal window: {len(chm_df)}")
        
        # Aggregate year statistics across all tiles
        all_year_counts = {}
        all_year_doy_combos = set()
        
        # Get from the footprints data
        for year in chm_df['year'].unique():
            year_chms = chm_df[chm_df['year'] == year]
            all_year_counts[int(year)] = len(year_chms)
            for doy in year_chms['doy'].unique():
                all_year_doy_combos.add((int(year), int(doy)))
        
        # Create summary statistics dictionary
        summary_stats = {
            'mosaic_name': os.path.basename(mosaic_vrt_path).replace('.vrt', ''),
            'target_year': target_year,
            'target_doy': target_doy,
            'delta_years': delta_years,
            'year_min': year_min,
            'year_max': year_max,
            'months_included': str(include_months) if include_months is not None else 'All',
            'grid_size_km': grid_size_km,
            'output_resolution_m': output_resolution,
            'n_tiles_total': len(results_df),
            'n_tiles_successful': len(successful_results),
            'n_tiles_failed': len(results_df) - len(successful_results),
            'n_chms_in_window': len(chm_df),
            'n_unique_years': len(all_year_counts),
            'n_unique_year_doy_combinations': len(all_year_doy_combos),
            'avg_coverage_percent_per_tile': float(successful_results['coverage_percent'].mean()),
            'min_coverage_percent': float(successful_results['coverage_percent'].min()),
            'max_coverage_percent': float(successful_results['coverage_percent'].max()),
            'total_chms_used': int(successful_results['n_images_used'].sum()),
            'avg_chms_per_tile': float(successful_results['n_images_used'].mean()),
            'aoi_area_km2': float(aoi_area_km2),
            'coverage_area_km2': float(coverage_area_km2),
            'aoi_coverage_proportion': float(aoi_coverage_proportion),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add year-specific CHM counts
        for year in sorted(all_year_counts.keys()):
            summary_stats[f'n_chms_year_{year}'] = all_year_counts[year]
        
        # Save summary CSV
        summary_csv_path = os.path.join(output_dir,
                                       f'mosaic_summary_{target_year}_DOY{target_doy:03d}_deltayr{delta_years:02d}_{months_str}.csv')
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Summary CSV saved: {summary_csv_path}")
        
        # Create detailed text summary
        summary_txt_path = os.path.join(output_dir,
                                       f'mosaic_summary_{target_year}_DOY{target_doy:03d}_deltayr{delta_years:02d}_{months_str}.txt')
        
        with open(summary_txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MOSAIC SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("MOSAIC PARAMETERS\n")
            f.write("-"*80 + "\n")
            f.write(f"Mosaic name:           {summary_stats['mosaic_name']}\n")
            f.write(f"Target year:           {target_year}\n")
            f.write(f"Target DOY:            {target_doy} ({datetime(target_year, 1, 1) + timedelta(days=target_doy-1):%B %d})\n")
            f.write(f"Year range:            {year_min} to {year_max} (±{delta_years} years)\n")
            f.write(f"Months included:       {summary_stats['months_included']}\n")
            f.write(f"Grid size:             {grid_size_km} km\n")
            f.write(f"Output resolution:     {output_resolution} m\n")
            f.write(f"Output directory:      {output_dir}\n")
            f.write(f"Generated:             {summary_stats['timestamp']}\n\n")
            
            f.write("TILE PROCESSING SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total tiles:           {summary_stats['n_tiles_total']}\n")
            f.write(f"Successful:            {summary_stats['n_tiles_successful']}\n")
            f.write(f"Failed:                {summary_stats['n_tiles_failed']}\n")
            f.write(f"Success rate:          {100*summary_stats['n_tiles_successful']/summary_stats['n_tiles_total']:.1f}%\n\n")
            
            f.write("AOI COVERAGE STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"AOI area:              {summary_stats['aoi_area_km2']:,.2f} km²\n")
            f.write(f"Coverage area:         {summary_stats['coverage_area_km2']:,.2f} km²\n")
            f.write(f"AOI coverage:          {summary_stats['aoi_coverage_proportion']*100:.2f}%\n\n")
            
            f.write("TILE COVERAGE STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Avg coverage per tile: {summary_stats['avg_coverage_percent_per_tile']:.2f}%\n")
            f.write(f"Min coverage per tile: {summary_stats['min_coverage_percent']:.2f}%\n")
            f.write(f"Max coverage per tile: {summary_stats['max_coverage_percent']:.2f}%\n\n")
            
            f.write("INPUT CHM SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total CHMs in window:  {summary_stats['n_chms_in_window']}\n")
            f.write(f"Total CHMs used:       {summary_stats['total_chms_used']}\n")
            f.write(f"Avg CHMs per tile:     {summary_stats['avg_chms_per_tile']:.1f}\n")
            f.write(f"Unique years:          {summary_stats['n_unique_years']}\n")
            f.write(f"Unique year-DOY combos: {summary_stats['n_unique_year_doy_combinations']}\n\n")
            
            f.write("CHMs BY YEAR\n")
            f.write("-"*80 + "\n")
            for year in sorted(all_year_counts.keys()):
                f.write(f"  {year}: {all_year_counts[year]:,} CHMs\n")
            f.write("\n")
            
            f.write("UNIQUE YEAR-DOY COMBINATIONS\n")
            f.write("-"*80 + "\n")
            for year, doy in sorted(all_year_doy_combos):
                date_obj = datetime(year, 1, 1) + timedelta(days=doy-1)
                n_chms = len(chm_df[(chm_df['year'] == year) & (chm_df['doy'] == doy)])
                f.write(f"  {year}-{doy:03d} ({date_obj.strftime('%Y-%m-%d')}): {n_chms} CHM(s)\n")
            f.write("\n")
            
            f.write("TEMPORAL DISTRIBUTION\n")
            f.write("-"*80 + "\n")
            f.write(f"Earliest CHM date:     {chm_df['date'].min()}\n")
            f.write(f"Latest CHM date:       {chm_df['date'].max()}\n")
            f.write(f"Date range span:       {(pd.to_datetime(chm_df['date'].max()) - pd.to_datetime(chm_df['date'].min())).days} days\n\n")
            
            f.write("OUTPUT FILES\n")
            f.write("-"*80 + "\n")
            f.write(f"Mosaic VRT:            {mosaic_vrt_path}\n")
            f.write(f"Processing results:    {results_path}\n")
            f.write(f"CHM list:              {chm_list_path}\n")
            f.write(f"Summary CSV:           {summary_csv_path}\n")
            f.write(f"Summary text:          {summary_txt_path}\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"Summary text saved: {summary_txt_path}")
        print(f"\nMosaic-level summary complete!")
        print(f"  {summary_stats['n_chms_in_window']} CHMs across {summary_stats['n_unique_years']} years")
        print(f"  {summary_stats['n_unique_year_doy_combinations']} unique year-DOY combinations")
        print(f"  AOI area: {summary_stats['aoi_area_km2']:,.2f} km²")
        print(f"  Coverage area: {summary_stats['coverage_area_km2']:,.2f} km²")
        print(f"  AOI coverage: {summary_stats['aoi_coverage_proportion']*100:.2f}%")
    
    else:
        print("  No successful tiles to summarize.")
    
    return results_df, mosaic_vrt_path
