#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import tensorflow as tf
import xarray as xr
from tqdm import trange
from functools import partial
#import tensorflow_addons as tfa
from skimage.measure import block_reduce
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa


from Models import ResNet, UNet3,PCNN, CNN

#Mask 

mask = np.load('mask.npy')
#mask = block_reduce(mask, block_size=(2, 2, 1), func=np.mean)
print(np.shape(mask))
mask = 1-mask

from test_utils import Test
timestep = 1
N_cycle = 1
k = 720
model =  UNet3( mask ,
                input_shape = (512,512,4*timestep + 6),
                kernel_size=3,
                activation = tfa.activations.mish,
                SE_prob = 0, 
                N_output = 1,
                n_filter = 32, 
                dropout_prob = 0)
path_to_save =  './Results/1Mars/UNet_lambda0_mu0_nu0/'


path_to_data = '../../data/data_HR_wo_sst/'

test = Test(model = model, mask = mask,k = k,  N_cycle = N_cycle, save_pred = True, timestep =timestep, path_to_save = path_to_save, path_to_data = path_to_data, noise = 0, noise_init = True)

fs, fs_pers, bias= test.test_model()
np.save(path_to_save +'clip_cycle_'+str(N_cycle)+ '_bias_mean.npy', bias)
np.save(path_to_save +'clip_cycle_'+str(N_cycle)+ '_fs_mean.npy', fs)
np.save(path_to_save +'clip_cycle_'+str(N_cycle)+ '_fs_mean_pers.npy', fs_pers)
