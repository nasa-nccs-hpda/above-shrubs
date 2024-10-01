import contextily as ctx
import shapely
import numpy as np
import folium

import rasterio
from rasterio.plot import show_hist, show
from rasterio.windows import Window

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
plt.rcParams.update({'font.size': 14})

def hillshade(array,azimuth,angle_altitude):
    #https://github.com/rveciana/introduccion-python-geoespacial/blob/master/hillshade.py
    azimuth = 360.0 - azimuth 

    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi/180.
    altituderad = angle_altitude*np.pi/180.

    shaded = np.sin(altituderad)*np.sin(slope) + np.cos(altituderad)*np.cos(slope)*np.cos((azimuthrad - np.pi/2.) - aspect)

    return 255*(shaded + 1)/2

def do_scalebar(ax):
    return ax.add_artist(ScaleBar(
        dx=1,
        units="m",
        dimension="si-length",
         length_fraction=0.15,
        scale_formatter=lambda value, unit: f' {value} m ',
        location='lower left'
        ))

def do_colorbar(ax, fig, image_hidden, label, SIZE='1.5%' ):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=SIZE, pad=0.15) #pad=-0.1)
    cb = fig.colorbar(image_hidden, cax=cax, orientation='vertical', extend='max')
    cb.set_label(label)
    
def rescale_pct_clip(array, pct=[1,80]):
    '''Re-scales data values of an array from 0-1 with percentiles'''
    array_min, array_max = np.nanpercentile(array,pct[0]), np.nanpercentile(array,pct[1])
    clip = (array - array_min) / (array_max - array_min)
    clip[clip>1]=1
    clip[clip<0]=0
    return clip

def rescale_multiband_for_plot(fn, rescaled_multiband_fn, bandlist = [4,3,2], pct=[5,95], nodata=-9999.0):
    
    # add a reduced res: https://gis.stackexchange.com/questions/434441/specifying-target-resolution-when-resampling-with-rasterio
    
    with rasterio.open(fn, "r+") as src1:
        
        src1.nodata = nodata
        arr_list = []
        for band in bandlist:
            arr = src1.read(band)
            arr_list.append(arr)
            
        with rasterio.open(rescaled_multiband_fn, 'w+',
                driver='GTiff',
                dtype= rasterio.float32,
                count=3,
                crs = src1.crs,
                width=src1.width,
                height=src1.height,
                transform=src1.transform,
                nodata=src1.nodata

            ) as dst:

            for i, band in enumerate(bandlist): 
                V = rescale_pct_clip(src1.read(band), pct=pct)
                dst.write(V,i+1)
            

class_name_list = ['Noise','Ground','Canopy','Top of canopy']
#class_color_list = ['lightgrey','brown','lightgreen','darkgreen']
class_color_list_ = ['lightgrey','brown','lightgreen','green']
HEIGHT_COLS_COLORS = ['#1b9e77','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d']

def subset_atl03(atl03_gdf, 
                 start=0, 
                 transect_length=int(1e4), 
                 class_color_list=['lightgrey','brown','lightgreen','green'], 
                 class_name_list=['Noise','Ground','Canopy','Top of canopy']
                ):
    '''transect_length is in number of photons
    '''
    
    finish = min(start + transect_length, atl03_gdf.shape[0])
    atl03_gdf_sub = atl03_gdf.iloc[start:finish]
    
    atl03_gdf_sub = classname_atl03(atl03_gdf_sub, class_color_list=class_color_list, class_name_list=class_name_list)
    
    return atl03_gdf_sub

def get_dist_vector(gdf, start, finish=-1):
    xsub = np.asarray(gdf['x'])[start:finish]
    ysub = np.asarray(gdf['y'])[start:finish]
    return np.sqrt((xsub-xsub[0])**2 + (ysub-ysub[0])**2)

def plot_atl03_transect(start=0, 
                        transect_length=50000, 
                        DO_HEIGHT=True, 
                        atl03_gdf=None, 
                        atl08_100m_gdf=None, 
                        atl08_30m_gdf=None, 
                        footprint_uav_gdf=None, 
                        ctx_kwargs=None, 
                        site=None, 
                        PLOT_UAV=False, 
                        HEIGHT_COLS=None
                       ):
    '''transect_length is in number of photons
    '''

    for beam in atl03_gdf.beam.unique():
        
        for doy in atl03_gdf.doy.unique():
        
            gdf = atl03_gdf[(atl03_gdf.beam == beam) & (atl03_gdf.doy == doy)]

            finish = min(start + transect_length, gdf.shape[0])
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(25, 5))
            
            ###
            # ATL03 photons
            #
            # ATL03 distance along x
            dist = get_dist_vector(gdf, start, finish)

            if DO_HEIGHT:
                elev = np.asarray(gdf['height'])[start:finish]
                if PLOT_UAV: elev_uav = np.asarray(gdf['height_uav'])[start:finish]
                if HEIGHT_COLS is not None:
                    ht_list = []
                    for i, ht_col in enumerate(HEIGHT_COLS):
                        ht_list.append( np.asarray(gdf[ht_col])[start:finish] )
                LIM_VAL_MAX, LIM_VAL_MIN = 10, 2
                YLAB = 'Height'
            else:
                elev = np.asarray(gdf['elev'])[start:finish]
                if PLOT_UAV: elev_uav = np.asarray(gdf['dsm_uav'])[start:finish]
                LIM_VAL_MAX, LIM_VAL_MIN = None, None
                YLAB = 'Elevation'
            phcl = np.asarray(gdf['class'])[start:finish]
            
            ###
            # Surfaces at ATL03 photons
            # UAV
            if PLOT_UAV:
                ax.scatter(dist, elev_uav, s=15, alpha=1, linewidth=0, c='k', label='UAV surface')
            
            # Other height (only) sources
            if DO_HEIGHT and HEIGHT_COLS is not None:
                for i, ht_col in enumerate(HEIGHT_COLS):
                    ax.scatter(dist, ht_list[i], s=5, alpha=1, linewidth=0, c=HEIGHT_COLS_COLORS[i], label=ht_col)
                
            for i in range( len(class_name_list) ):
                idx = (phcl == i)
                if np.count_nonzero(idx) > 0:
                    ax.scatter(dist[idx], elev[idx], s=15, alpha=0.5, linewidth=0, c=class_color_list[i], label=class_name_list[i])
            if LIM_VAL_MAX is not None:
                ax.set_ylim([np.median(elev)-LIM_VAL_MIN, np.median(elev)+LIM_VAL_MAX])
            ###
            # ATL08 100m
            #
            if atl08_100m_gdf is not None:
                atl08_100m_gdf_beam = atl08_100m_gdf[(atl08_100m_gdf.beam == beam) & (atl08_100m_gdf.doy == doy)]
                 # ATL08 distance along x
                dist_atl08 = get_dist_vector(atl08_100m_gdf_beam, 0) # plot all atl08
                elev_atl08 = np.asarray(atl08_100m_gdf_beam['h_te_median'])[0:len(dist_atl08)]
                if DO_HEIGHT:
                    elev_atl08 = np.asarray(atl08_100m_gdf_beam['h_canopy'])[0:len(dist_atl08)]
                # Add to plot
                ax.scatter(dist_atl08, elev_atl08, s=100, alpha=1, linewidth=0, color='orange', label='ATL08 100m') #'#bfcb0f'

            ###
            # ATL08 30m
            #
            if atl08_30m_gdf is not None:
                atl08_30m_gdf_doy = atl08_30m_gdf[ (atl08_30m_gdf.doy == doy)]
                if atl08_30m_gdf_doy.shape[0] > 0:
                    # ATL08 distance along x
                    dist_atl08_30m = get_dist_vector(atl08_30m_gdf_doy, 0) # plot all atl08
                    elev_atl08_30m = np.asarray(atl08_30m_gdf_doy['h_te_median'])[0:len(dist_atl08_30m)]
                    if DO_HEIGHT:
                        elev_atl08_30m = np.asarray(atl08_30m_gdf_doy['h_canopy'])[0:len(dist_atl08_30m)]
                    # Add to plot
                    ax.scatter(dist_atl08_30m, elev_atl08_30m, s=50, alpha=1, linewidth=0, color='red', label='ATL08 30m')
                    
            if footprint_uav_gdf is not None:
                ax2 = footprint_uav_gdf.to_crs(gdf.crs).plot(ax=ax2, facecolor='gray')
            ax2 = gdf.iloc[start:finish].plot(ax=ax2, alpha=0.25, c=gdf['color'][start:finish], label=gdf['class_name'][start:finish])#, markersize=gdf['elev'][start:finish])
            if atl08_100m_gdf is not None:
                atl08_100m_gdf_beam.plot(ax=ax2, alpha=1, edgecolor='k', facecolor='none')
                atl08_100m_gdf_beam.geometry.centroid.plot(ax=ax2, alpha=1, edgecolor='orange', facecolor='none')
            if atl08_30m_gdf is not None:
                if atl08_30m_gdf_doy.shape[0] > 0:
                    atl08_30m_gdf_doy.plot(ax=ax2, alpha=1, edgecolor='red', facecolor='none')
                    atl08_30m_gdf_doy.geometry.centroid.plot(ax=ax2, alpha=1, edgecolor='red', facecolor='none')
            #ax2 = ctx.add_basemap(ax=ax2, zoom=18, **ctx_kwargs)

            ax.legend(loc='upper right')
            lon_start = np.asarray(gdf['lon'])[start]
            lat_start = np.asarray(gdf['lat'])[start]
            title = f"Site ({site}) start coords: (Lon) {lon_start:.4f} (Lat) {lat_start:.4f}\nBeam: {[beam.upper() for beam in gdf.beam.unique()]}, Yr: {[yr for yr in gdf.year.unique()]}, DOY: {[doy for doy in gdf.doy.unique()]}"
            ax.set_title(title, fontsize=8, loc='left')
            ax.set(xlabel='Distance (m)', ylabel=f'{YLAB} (m)')

            fig.suptitle(f"ICESat-2/ATLAS vegetation measurements", fontsize=18)
            fig.canvas.draw()
            plt.show()
