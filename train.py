#!/usr/bin/env python3


from functools import partial
import numpy as np
import numpy.ma as ma
from skimage.measure import block_reduce


import matplotlib.pyplot as plt
import datetime

import tensorflow.keras.utils as conv_utils
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import InputSpec, Concatenate
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    UpSampling2D,
    Dropout,
    LeakyReLU,
    BatchNormalization,
    Activation,
    Lambda,
)

tf.keras.backend.set_floatx("float32")
from Models import *
from Utils2 import *


mask = np.load("mask_HR.npy")
mask = block_reduce(mask, block_size=(2, 2, 1), func=np.min)
print(np.shape(mask))
mask = 1 - mask

path = "./Results/22Fev23/UNet_lambda0_mu0/"
train_model(
    UNet3,
    mask,
    path_to_save=path + "/",
    path_to_data="../../data/data_wo_sst/",
    batch_size=32,
    kernel_size=3,
    epochs=500,
    verbose=True,
    retrain=False,
    new_lambda=100,
    N_output=1,
    input_shape=(256, 256, 10),
    lambda_=0,
    mu=0,
    lambda_scheme="False",
    timestep=1,
    learning_rate=5e-5,
    dropout_prob=0,
    SE_prob=0,
    n_filters=16,
    activation=tfa.activations.mish,
    loss="sum",
)
