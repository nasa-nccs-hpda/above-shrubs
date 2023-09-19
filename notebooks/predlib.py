#! /usr/bin/env python

#Library containing various functions for the stacking and validation of SR-lite based canopy height model predictions

import os, sys, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append('/home/pmontesa/code/pygeotools')
from pygeotools.lib import *

def gdal_merge_gtiff_stack(out_fn, fn_sr, fn_warp_dtm, GDAL_MERGE_PATH = '/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/envs/ilab-tensorflow/bin'):
    
    gdal_cmd = [f"{os.path.join(GDAL_MERGE_PATH, 'gdal_merge.py')}", "-o", out_fn, "-separate", "-of", "gtiff", fn_sr, fn_warp_dtm]
    print(gdal_cmd)
    proc = subprocess.Popen(gdal_cmd, stdout = subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = proc.communicate()
    print(out)
    return proc.returncode, out_fn

def do_pred_stack(fn_sr, dtm_path, outdir):

    GDAL_MERGE_PATH = '/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/envs/ilab-tensorflow/bin/'
    
    f = os.path.basename(fn_sr).replace('.tif','')
    print(f)
    
    outdir_stack = os.path.join(outdir, f)
    if not os.path.exists(outdir_stack): os.makedirs(outdir_stack)
    
    # Warp the DTM to the SRlite
    warp_ds_list = warplib.diskwarp_multi_fn([fn_sr, dtm_path], res=fn_sr, extent=fn_sr, t_srs=fn_sr, r='bicubic', dst_ndv=-10001, outdir=outdir_stack, verbose=False)
    
    # Need to apply common mask before saving
    
    # Get the full path of the warped DTM
    fn_warp_dtm = os.path.join(outdir, f, os.path.basename(dtm_path).replace('.tif', '_warp.tif'))
    #print(fn_warp_dtm)
    
    out_fn = os.path.join(outdir,f,f + '_pred_stack.tif')
                                
    # Works, but only in notebook                         
    #command = f"{GDAL_MERGE_PATH}gdal_merge.py -o {out_fn} -separate -of gtiff {fn_sr} {fn_warp_dtm}"
    #!eval $command
                                
    exit_code, out_fn = gdal_merge_gtiff_stack(out_fn, fn_sr, fn_warp_dtm)
    os.remove(fn_warp_dtm)  
    
    if exit_code != 0:
        print(f"gdal_merge_gtiff_stack failed with a non-zero exit code: {exit_code}")
        exit(exit_code)
    
    return out_fn

def validate_chm_pred(pred_fn, lvis_fn, index_of_scaling = 0, scale_factor=0.1, CHM_LIMS=(0,15), TCC_LIMS=(0,100), RES=30, SRC_NAME='CNN CHM', RETURN_DF=False, DST_NDV=-9999, DEBUG=False):
    
    '''Validate a predicted CHM with LVIS'''
    
    # This order is important
    lvis_tcc_fn = lvis_fn.replace('_RH098_mean_30m.tif','_CC_gte_01p37m_30m.tif')
    fn_list = [pred_fn, lvis_fn, lvis_tcc_fn]
    
    if DEBUG: [print(os.path.basename(fn)) for fn in fn_list]

    
    # ---Doing the warp ---
    # Here, the warped datasets and their arrays in these lists will be in the same order as your initial filenames list (fn_list)
    warp_ds_list = warplib.memwarp_multi_fn(fn_list, res=RES, extent='intersection', t_srs=lvis_fn, r='near', dst_ndv=DST_NDV, verbose=False)
    warp_ma_list = [iolib.ds_getma(ds) for ds in warp_ds_list]
    
    if DEBUG: print(f'Length of warp ma list: {len(warp_ma_list)}')
    
    # Scale preds in decimeters to meters
    for i, ma in enumerate(warp_ma_list): 
        if DEBUG: print(f'Warped ma min, max: {np.nanmin(ma)}, {np.nanmax(ma)}')
        if i == index_of_scaling: 
            #print(f'Mean (orig): {warp_ma_list[i].mean()}')
            warp_ma_list[i] = ma * scale_factor
            #print(f'Mean (scaled): {warp_ma_list[i].mean()}')
            
    # Common mask
    ##print('Common mask: masking negative and 0 values in each input...')
    #warp_valid_ma_list = [ np.ma.masked_where(ma <= 0, np.ma.masked_invalid(ma)) for ma in warp_ma_list]
    warp_valid_ma_list = [ np.ma.masked_where(ma == DST_NDV, np.ma.masked_invalid(ma)) for ma in warp_ma_list]
    common_mask = malib.common_mask(warp_valid_ma_list)
    warp_ma_masked_list = [np.ma.array(ma, mask=common_mask) for ma in warp_ma_list]
    
    if DEBUG: [print(f'Warped ma masked min, max: {np.nanmin(ma)}, {np.nanmax(ma)}') for i, ma in enumerate(warp_ma_masked_list)]
        
    if RETURN_DF:
        return build_pred_obs(warp_ma_masked_list, RETURN_DF=True)
    else:
        # Do plots
        plot_maps(warp_ma_masked_list[0:2], fn_list, clim_list = [CHM_LIMS, CHM_LIMS, TCC_LIMS], title_text = '', map_label='Height [m]', figsize=(12,3))
        plot_hists(warp_ma_masked_list[0:2], fn_list,  title_text = '', figsize=(12,3)),
        plot_scatter(warp_ma_masked_list[0:2], SRC_NAME, RES, (5,5))
        plot_diff_map(warp_ma_masked_list[0:2], SRC_NAME, clim_list = [(-10,10)])
    
def plot_maps(masked_array_list, names_list, figsize=None, cmap_list=None, clim_list=None, title_text="", map_label='Reflectance (%)', COLORBAR_EXTEND_DIR='max'):
    
    if figsize is None:
        figsize = (len(names_list) * 7,5)
    
    fig, axa = plt.subplots( nrows=1, ncols=len(masked_array_list), figsize=figsize, sharex=False, sharey=False)

    for i, ma in enumerate(masked_array_list):
        
        if cmap_list is None:
            cmap = 'magma'
            # For TCC, in position 3
            if i == 2:
                cmap = 'viridis'
        else:
            cmap = cmap_list[i]

        if clim_list is None:
            clim = calcperc_nan(ma, perc=(1,95))
        else:
            clim = clim_list[i]

        f_name = names_list[i]
        
        if len(masked_array_list) == 1:
            axa = [axa]
        
        divider = make_axes_locatable(axa[i])
        cax = divider.append_axes('right', size='2.5%', pad=0.05)
        im1 = axa[i].imshow(ma, cmap=cmap , clim=clim )
        cb = fig.colorbar(im1, cax=cax, orientation='vertical', extend=COLORBAR_EXTEND_DIR)
        axa[i].set_title(title_text + os.path.split(f_name)[1], fontsize=10)
        cb.set_label(map_label)

        plt.tight_layout() 

def plot_hists(masked_array_list, names_list, figsize=None, title_text="", map_label='Reflectance (%)'):
    
    if figsize is None:
        figsize = (len(names_list) * 7,5)
    
    fig, axa = plt.subplots(nrows=1, ncols=len(masked_array_list), figsize=figsize, sharex=False, sharey=False)

    for i, ma in enumerate(masked_array_list):
        f_name = names_list[i]
        print(f"\n{ma.count()} valid pixels in INPUT MASKED ARRAY version of {f_name}")

        h = axa[i].hist(ma.compressed(), bins=256, alpha=0.75, facecolor='gray')
        axa[i].set_title(title_text + os.path.split(f_name)[1], fontsize=10)

    plt.tight_layout() 
    
def calcperc_nan(b, perc=(0.1,99.9)):
    """Calculate values at specified percentiles
    """
    b = malib.checkma(b)
    if b.count() > 0:
        low = np.nanpercentile(b.compressed(), perc[0])
        high = np.nanpercentile(b.compressed(), perc[1])
    else:
        low = 0
        high = 0
    return low, high

def set_ndv_to_nan(ma, ndv):
    ma[ma == -ndv] = np.nan
    return ma

def get_ndv(r_fn):
    with rasterio.open(r_fn) as src:
        return src.profile['nodata']
    
def map_image_band(cog_fn, band_num=1, vmin=0.20, vmax=0.45, figsize=(5, 5)):
    
    with rasterio.open(cog_fn) as dataset:

        fig, ax = plt.subplots(figsize=figsize)

        # use imshow so that we have something to map the colorbar to
        image_hidden = ax.imshow(dataset.read(band_num), 
                                 cmap='nipy_spectral', 
                                 vmin=vmin, 
                                 vmax=vmax)

        # plot on the same axis with rio.plot.show
        image = show(dataset.read(band_num), 
                              transform=dataset.transform, 
                              ax=ax, 
                              cmap='nipy_spectral', 
                              vmin=vmin, 
                              vmax=vmax)

        # add colorbar using the now hidden image
        fig.colorbar(image_hidden, ax=ax)
        
def tcc_classifier(row):
    if row["tcc_ref"] > 0 and row["tcc_ref"] <= 0.2 * 1e4:
        return "0-20%"
    elif row["tcc_ref"] > 0.2 * 1e4 and row["tcc_ref"] <= 0.4 * 1e4:
        return "21-40%"
    elif row["tcc_ref"] > 0.4 * 1e4 and row["tcc_ref"] <= 0.6 * 1e4:
        return "41-60%"
    elif row["tcc_ref"] > 0.6 * 1e4 and row["tcc_ref"] <= 0.8 * 1e4:
        return "61-80%"
    else:
        #row["tcc_ref"] > 0.8 * 1e4 and row["tcc_ref"] <= 1.0 * 1e4:
        return "81-100%"
        
def build_pred_obs(warp_ma_masked_list, RETURN_DF=False):
    
    # Prep the x and y data
    # Reference is first element in list, which needs to be the y-var: b/c we are predicting SR from TOA
    if len(warp_ma_masked_list) == 3:
        src_y_var, ref_x_var, tcc_x_var = [ma.ravel() for ma in warp_ma_masked_list]
    if len(warp_ma_masked_list) == 2:
        src_y_var, ref_x_var = [ma.ravel() for ma in warp_ma_masked_list]
    
    y_var = src_y_var[src_y_var.mask == False]
    x_var = ref_x_var[ref_x_var.mask == False]
    
    if len(warp_ma_masked_list) == 2:
        tcc_x_var = x_var.copy()
        tcc_x_var[:] = np.nan 
    
    #if len(warp_ma_masked_list) == 3:
    tcc = tcc_x_var[tcc_x_var.mask == False]
    
    if RETURN_DF:
        df = pd.DataFrame(data = np.transpose([y_var,x_var,tcc]), columns=['ht_m_src','ht_m_ref','tcc_ref'])
        df['tcc_class'] = df.apply(tcc_classifier, axis=1)
        return df
    else:
        return y_var, x_var, tcc
        
def plot_scatter(warp_ma_masked_list, SRC_NAME, RES, FIG_SIZE, RETURN_DF=False):
    
    y_var, x_var, tcc_var = build_pred_obs(warp_ma_masked_list, RETURN_DF=RETURN_DF)
    
    figsize=(6,6)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=FIG_SIZE, sharex=True, sharey=True)

    hb = ax.hexbin(x_var.ravel().data, y_var.ravel().data, gridsize=500, 
                   bins='log',
                   cmap='cividis')
    ax.axis([x_var.min(), x_var.max(), y_var.min(), y_var.max()])
    ax.set_title("Canopy Height Map Validation")
    plt.xlabel("Canopy Height [m]\nreference lidar")
    plt.ylabel(f"Canopy Height [m]\n{SRC_NAME}")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(f'# {RES} m pixel obs.')
    
    if RETURN_DF:
        
        return df_ht_val
    
def plot_diff_map(warp_ma_masked_list, SRC_NAME, clim_list = [(-10,10)]):
    
    # Note: src array should be the first in the list; reference array second
    # Difference Map
    diff_ma = warp_ma_masked_list[0]-warp_ma_masked_list[1]

    if False:
        print('Diff array mins/maxs')
        print(f'Diff ma min: {np.nanmin(diff_ma)}')
        print(f'Diff ma max: {np.nanmax(diff_ma)}') 

    plot_maps([diff_ma], [''], title_text=f'Difference: {SRC_NAME} - reference', clim_list = clim_list, 
              cmap_list=['RdBu'],
              # This is setup to correspond with src - ref differencing
              map_label = 'Difference [m]\n under-estimate <---> over-estimate',
              COLORBAR_EXTEND_DIR='both'
             )
    
def do_validation_df(IMG_IDX, list_of_pred_fn, list_of_ref_fn, out_csv_dir, DEBUG=False):
    
    #print(os.path.basename(list_of_pred_fn[IMG_IDX]))
    
    pred_file = os.path.basename(list_of_pred_fn[IMG_IDX])
    ref_file = os.path.basename(list_of_ref_fn[IMG_IDX])
    out_csv_fn = os.path.join(out_csv_dir, pred_file.replace('.tif','') + '_' + ref_file.replace('.tif','') + f'_val{IMG_IDX:04}.csv')
    
    if not os.path.isfile(out_csv_fn): 
        
        print(f'Run {IMG_IDX}')
        
        df = validate_chm_pred(list_of_pred_fn[IMG_IDX], list_of_ref_fn[IMG_IDX], scale_factor=0.1, RETURN_DF=True, DEBUG=DEBUG)
        df['file'] = pred_file
        
        if DEBUG: print(f'Head before attributes: {df.head()}')
    
        # Write fields
        df = footprintlib.get_attributes_from_filename(df, 'CHM pred', '-sr', DROP_FILE_DUPLICATES=False)
        
        if DEBUG: print(f'Head after attributes: {df.head()}')
    
        # Write df (b/c returning all to a list in memory isnt working...)
        df.to_csv(out_csv_fn)
    else:
        print(f'Skip {IMG_IDX}')
    
    #return df  