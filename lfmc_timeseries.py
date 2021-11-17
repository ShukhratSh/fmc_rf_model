
import sys
sys.path.append('/g/data/xc0/user/Shukhrat/dea-notebooks/Scripts/') # i.e. dea-notebooks/Scripts/

import datacube
from datacube.utils import geometry
from dea_datahandling import load_ard
from datacube.storage.masking import make_mask
from datacube.storage.masking import mask_invalid_data
import rasterio.crs
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import time
from datetime import date
from datetime import datetime


def sentinel_timeseries(df, file_path:str, begin:str = None, end:str = None):

    """This function takes area of interest from geometry of a geopandas dataframe, 
    and saves a xarray dataset of Fuel Moisture Content.

    Keyword arguments:
    df - gpd.Dataframe, shapefile or other file opened in geopandas
    begin - str, date of beginning of timeseries in format YYYY-MM-DD)
    end - str, date of ending of timeseries in format YYYY-MM-DD)
    file_path - str, absolute filepath to save file to e.g. '/g/data/..'
    """

    if type(begin) is str:
        None
    else:
        begin = '2015-01-01'

    if type(end) is str:
        None
    else:
        end = '2020-12-31'

    # get bounds and crs from dataframe (minx, miny, maxx, maxy)
    bounds = df.geometry.bounds
    crs = df.crs.to_epsg()

    dc = datacube.Datacube(app='fmc')

    # start loop through df of bounds, updating query with y and x points
    for idx_name,series in bounds.iterrows():
        print(f'Analysing {idx_name} data from {begin} to {end}')
        minx,miny,maxx,maxy = series

        query = {
            'y': (miny,maxy),
            'x': (minx,maxx),
            'crs': f'EPSG:{crs}',
            'output_crs': 'EPSG:3577',
            'resolution': (-25, 25),
            'time': (begin,end),
            'measurements': ["nbar_blue","nbar_green","nbar_red",
                             "nbart_red_edge_1","nbart_red_edge_2","nbart_red_edge_3",
                             "nbar_nir_1","nbar_nir_2","nbar_swir_2","nbar_swir_3",'fmask'
                            ],
            'group_by':'solar_day',
            'min_gooddata':0.5
        }

        s2_ds = load_ard(dc=dc,
                     products=['s2a_ard_granule', 's2b_ard_granule' ],
                     **query)

    ## start FMC model

        refl = s2_ds[["nbar_green","nbar_red","nbart_red_edge_1","nbart_red_edge_2","nbart_red_edge_3",
                 "nbar_nir_1","nbar_nir_2","nbar_swir_2","nbar_swir_3"]].to_array().values/10000

        ndvi=((s2_ds.nbar_nir_1-s2_ds.nbar_red)/(s2_ds.nbar_nir_1+s2_ds.nbar_red)).values 
        s2_ds['ndii']=((s2_ds.nbar_nir_1-s2_ds.nbar_swir_2)/(s2_ds.nbar_nir_1+s2_ds.nbar_swir_2))
        ndii=s2_ds.ndii.values

        refl = np.concatenate([refl,ndii[None,:,:]], axis=0)

        print(f'Shape of reflectance; {refl.shape}, and NDVI data; {ndvi.shape}')


        ds = xr.open_dataset("/g/data/ub8/au/LandCover/OzWALD_LC/VegH_2007-2010_mosaic_AustAlb_25m.nc")
        vegh = ds.VegH.sel(x=s2_ds.x, y=s2_ds.y)

        ds = xr.open_dataset("/g/data/ub8/au/LandCover/OzWALD_LC/WCF_2018_mosaic_AustAlb_25m.nc")
	# fillmissmatch
        wcf = ds.WCF.sel(x=s2_ds.x, y=s2_ds.y)

        """
        grass = (wcf<10)*(vegh<2)*(vegh!=0)*1
        shrub = (wcf>=10)*(vegh<2)*(vegh!=0)*2
        sav_2 = (wcf>=10)*(wcf<20)*(vegh>=2)*3
        sav_1 = (wcf>=20)*(wcf<30)*(vegh>=2)*4
        forest = (wcf>=30)*(vegh>=2)*5
        """

        # Original classes with savanah merged into shrub
        grass = (wcf<10)*(vegh<2)*(vegh!=0)*1
        shrub = (wcf>=10)*(vegh<2)*(vegh!=0)*2
        sav_2 = (wcf>=10)*(wcf<20)*(vegh>=2)*2
        sav_1 = (wcf>=20)*(wcf<30)*(vegh>=2)*2
        forest = (wcf>=30)*(vegh>=2)*3

        mask = shrub+grass+sav_1+sav_2+forest
        mask = mask.values.T


        # read Look Up Table from file
        df = pd.read_csv(file_path+'LUT_S2.csv', index_col='ID')
        #NDII=nbar_nir_1-nbar_swir_2/nbar_nir_1+nbar_swir_2
        df = df.drop(columns=['lai','soil','n','443','490','1375','945'])
        df.columns = ['fmc','landcover','green','red','red_edge1','red_edge2','red_edge3','nir1','nir2','swir2','swir3']
        # df[df.landcover=='forest'].shape, df[df.landcover=='shrub'].shape, df[df.landcover=='grass'].shape
        df['ndii'] = (df['nir1']-df['swir2'])/(df['nir1']+df['swir2'])

        canvas1 = np.ones(ndvi.shape, dtype=np.float32) * np.nan
        top_n = 40

        lut_map = {
            1: df[df.landcover == "grass"][
                [
                    "fmc",
                    "green",
                    "red",
                    "red_edge1",
                    "red_edge2",
                    "red_edge3",
                    "nir1",
                    "nir2",
                    "swir2",
                    "swir3",
                    "ndii",
                ]
            ].values,
            2: df[df.landcover == "shrub"][
                [
                    "fmc",
                    "green",
                    "red",
                    "red_edge1",
                    "red_edge2",
                    "red_edge3",
                    "nir1",
                    "nir2",
                    "swir2",
                    "swir3",
                    "ndii",
                ]
            ].values,
            3: df[df.landcover == "forest"][
                [
                    "fmc",
                    "green",
                    "red",
                    "red_edge1",
                    "red_edge2",
                    "red_edge3",
                    "nir1",
                    "nir2",
                    "swir2",
                    "swir3",
                    "ndii",
                ]
            ].values,
        }

        # Add the squares of the LUT entries to speed up computation inside loop
        lut_map[4] = np.einsum("ij,ij->i", lut_map[1][:, 1:], lut_map[1][:, 1:]) ** 0.5
        lut_map[5] = np.einsum("ij,ij->i", lut_map[2][:, 1:], lut_map[2][:, 1:]) ** 0.5
        lut_map[6] = np.einsum("ij,ij->i", lut_map[3][:, 1:], lut_map[3][:, 1:]) ** 0.5

        for t in range(ndvi.shape[0]):

            for j in range(mask.shape[0]):

                for i in range(mask.shape[1]):
                    x = refl[:,t, j, i]
                    m = mask[j, i]

                    if m == 0 or ndvi[t, j, i] < 0.15:
                        continue

                    θ = -1 * (
                        np.einsum("ij,j->i", lut_map[m][:, 1:], x)
                        / (np.einsum("i,i->", x, x) ** 0.5 * lut_map[m + 3])
                    )

                    idxs = np.argpartition(θ, top_n)[:top_n]
                    canvas1[t, j, i] = np.median(lut_map[m][idxs, 0])

        #s2_ds['FMC'] = (['time','y','x'], canvas1)
        #s2_ds['FMC'] = s2_ds['FMC'] * s2_ds.fmask.where(s2_ds.fmask==1,np.NaN)
        clear_mask = make_mask(s2_ds.fmask, fmask='valid')
		s2_ds['FMC'] = s2_ds['FMC'].where(clear_mask)
		s2_ds['Forest'] = forest
        s2_ds['Grass'] = grass
        s2_ds['Shrub'] = shrub
        # drop variables not to be saved in netcdf
        s2_ds = s2_ds.drop_vars(['nbar_blue','nbar_green','nbar_red','nbart_red_edge_1','nbart_red_edge_2',
                       'nbart_red_edge_3','nbar_nir_1','nbar_nir_2','nbar_swir_2','nbar_swir_3','ndii','fmask'])

        for variable in s2_ds.variables.values():
            variable.attrs = {}
        s2_ds.attrs['units'] = '% dry matter'

        print("Saving out to:",file_path+idx_name+'.nc')
        s2_ds.to_netcdf(file_path+idx_name+'.nc',mode='w')
        print('------------')

    return



# Set path names, load the shapefile, modify shapefile dataframe
#file_path = '/home/147/ss8137/dea-notebooks/fmc/lfmc_timeseries-main/
path_file = '/g/data/xc0/user/Shukhrat/scripts/ml_fmc_sentinel/' # change this to accessible storage
shapefile_path = '/g/data/xc0/user/Shukhrat/shp_files/ACT_OrroralValley_2020/orroral_fire.shp'
df = gpd.read_file(shapefile_path)
df.set_index('title', inplace=True) # used to name the netcdf files
#df = df.to_crs("EPSG:3577")
#df['geometry'] =df.geometry.buffer(52)
#df.plot()
#df.crs # check to see if dataframe has a coordinate reference system and is appropriate, otherwise df.set_crs(epsg=)

sentinel_timeseries(df, path_file,'2015', '2020')




