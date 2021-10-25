#!/usr/bin/env python
# coding: utf-8

# ### 1.- Download and load the Random Forest model.

# In[2]:


# This connection is blocked on the DEA Sandbox, so needs to be downloaded locally and uploaded to Sandbox manually
#!wget https://drive.google.com/file/d/1jfEEYh6wH25tW0InGPgx8aMXt8Cg-dUh/view?usp=sharing

#pip3 install pickle5
import sys
sys.path.append('/home/147/ss8137/.local/lib/python3.9/site-packages/')
import pickle5 as pickle
import sklearn

with open('/g/data/xc0/user/Shukhrat/scripts/sentinel2_fmc-main/rf_fmc.pickle', 'rb') as handle:
    rf = pickle.load(handle)
    print(rf)


# ### 2.- Load DEA data for Namadgi region

# In[2]:

sys.path.append('/g/data/xc0/user/Shukhrat/dea-notebooks/Scripts')
from dea_datahandling import load_ard
import matplotlib.pyplot as plt
import datacube
from datacube.storage.masking import make_mask
#from datacube.storage.masking import mask_invalid_data
import xarray as xr
import numpy as np


dc = datacube.Datacube(app='fmc')

query = {
        'y': (-4019984.18, -4116695.952),
        'x': (1561223.824, 1630941.803),
        'crs': 'EPSG:3577',
        'output_crs': 'EPSG:3577',
        'resolution': (-25, 25),
        'time': ('2015-01-01', '2020-12-31'),
        'measurements': ["nbar_blue","nbar_green","nbar_red",
                         "nbart_red_edge_1","nbart_red_edge_2","nbart_red_edge_3",
                         "nbar_nir_1","nbar_nir_2",
                         "nbar_swir_2","nbar_swir_3","fmask" ],
	'group_by':'solar_day',
	'min_gooddata':0.5}

ds = load_ard(dc=dc, products=['s2a_ard_granule', 's2b_ard_granule'], **query)

#ds = ds.isel(time=0)

#ds[['nbar_red', 'nbar_green', 'nbar_blue']].to_array().plot.imshow(robust=rue, figsize=(8,8))


# ### 3.- Add NDVI and NDII normalised indices to the dataset

# In[3]:


ds['ndvi']=((ds.nbar_nir_1-ds.nbar_red)/(ds.nbar_nir_1+ds.nbar_red))
ds['ndii']=((ds.nbar_nir_1-ds.nbar_swir_2)/(ds.nbar_nir_1+ds.nbar_swir_2))


# ### 4.- Stack and reshape dataset to be compatible with the RF input

# In[4]:


refl = ds[['ndvi','ndii','nbar_red','nbar_green','nbar_blue','nbar_nir_1','nbar_nir_2','nbar_swir_2','nbar_swir_3']].to_array().values
refl_rf = refl.reshape((9,-1)).swapaxes(0,1)

import numpy as np
refl_rf = np.nan_to_num(refl_rf, copy=False, posinf=np.nan, neginf=np.nan) # Converting infinite values to nans

rf_fmc = rf.predict(refl_rf)

fmc = rf_fmc.reshape(refl.shape[1:])


### 7. Creating FMC xarray dataset
# extracting coordinates from original data to add to the predicted data
x = ds.x.values
print(x)
y = ds.y.values
print(y)
time = ds.time.values
print(time)

# Creating a xarray dataset
ds_fmc = xr.Dataset(
    data_vars=dict(
        fmc=(["time", "y", "x"], fmc),
    ),
    coords=dict(
        x=(["x"], x),
        y=(["y"], y),
        time=time,
    ),
    attrs=dict(description="RF derived FMC"),
)

### 8. Remove cloud areas from image as they represent false positives

# Create the mask based on "valid" pixels
clear_mask = make_mask(ds.fmask, fmask="valid")

# Apply the mask
ds_fmc_clear = ds_fmc.where(clear_mask)

ds_fmc_clear.to_netcdf('/g/data/xc0/user/Shukhrat/scripts/ml_fmc_sentinel/ml_fmc_badja_2015_2020.nc',mode='w')





