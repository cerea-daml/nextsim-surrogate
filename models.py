import numpy as np
import numpy.ma as ma
import tensorflow_addons as tfa
import tensorflow.keras.utils as conv_utils
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import InputSpec, Concatenate
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation, Lambda
tf.keras.backend.set_floatx('float32')



class PConv(layers.Conv2D):
    ''' Implementation of Partial 2D Convolution as based on the article Image Inpainting for Irregular Holes Using Partial Convolutions from Liu & al.
                Input : mask - an array of the same shape as the images which will be put at the input of the neural network,
                mask indicates 0 when the pixel is hidden and 1 when it is visible'''

    def __init__(self,mask,over_compensation,*args, n_channels=2, mono=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim=4)]
        self.mask = tf.cast(mask,'float32', name='mask')
        self.oc = over_compensation
    def build(self, input_shape):
        """Adapted from original _Conv() layer of Keras
        param input_shape: list of dimensions for [img, mask]
        """

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')

        self.input_dim = input_shape[channel_axis]

        # Image kernel
        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='img_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)


        # Calculate padding size to achieve zero-padding
        self.pconv_padding = (
            (int((self.kernel_size[0])/2), int((self.kernel_size[0])/2)),
            (int((self.kernel_size[0])/2), int((self.kernel_size[0])/2)),
        )


        # Mask kernel
        kernel_mask = K.ones(shape=self.kernel_size + (1, 1))
        self.padded_mask = K.spatial_2d_padding(self.mask, self.pconv_padding, self.data_format)
        valid_pixels = K.conv2d(
            self.padded_mask, kernel_mask,
            strides=1,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Window size - used for normalization
        window_size = self.kernel_size[0] * self.kernel_size[1]
        if self.oc :
            self.mask_ratio = window_size * self.mask / (valid_pixels+1E-8)
        else :
            self.mask_ratio = self.mask
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        '''
        We will be using the Keras conv2d method, and essentially we have
        to do here is multiply the mask with the input X, before we apply the
        convolutions. For the mask itself, we apply convolutions with all weights
        set to 1.
        Subsequently, we clip mask values to between 0 and 1
        '''

        # Both image and mask must be supplied

        # Padding done explicitly so that padding becomes part of the masked partial convolution
        images = K.spatial_2d_padding(inputs, self.pconv_padding, self.data_format)

        # Apply convolutions to image

        img_output = K.conv2d(
            (images*self.padded_mask), self.kernel,
            strides=1,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Normalize image output
        img_output = img_output * self.mask_ratio

        # Apply bias only to the image (if chosen to do so)
        if self.use_bias:
            img_output = K.bias_add(
                img_output,
                self.bias,
                data_format=self.data_format)

        # Apply activations on the image
        if self.activation is not None:
            img_output = self.activation(img_output)

        return img_output



def encoder_block (prev_layer, mask, n_filters,kernel_size = 3,over_compensation = True,  activation = 'relu', kernel_initializer='HeNormal',SE_prob = 0, dropout_prob = 0.2, batch_norm = True) :

    ''' Enocder block of UNet using partial convolution'''

    conv = PConv(filters = n_filters,
                    mask = mask,
                    over_compensation = over_compensation,
                    kernel_size = kernel_size,
                    padding = 'same',
                    kernel_initializer = kernel_initializer,
                    activation = activation)(prev_layer)
    if SE_prob >0 :
        nb_chan = keras.backend.int_shape(conv)[-1]
        mp = layers.GlobalAveragePooling2D() (conv)
        mp = layers.Dense(nb_chan // SE_prob, activation = 'relu') (mp)
        mp = layers.Dense(nb_chan, activation = 'sigmoid') (mp)
        conv = layers.Multiply () ([conv,mp])
    if dropout_prob > 0:
        conv = layers.Dropout(dropout_prob)(conv)

    conv = PConv(filters = n_filters,
                    mask = mask,
                    over_compensation = over_compensation,
                    kernel_size = kernel_size,
                    padding = 'same',
                    kernel_initializer = kernel_initializer,
                    activation = activation)(conv)
    if SE_prob >0 :
        nb_chan = keras.backend.int_shape(conv)[-1]
        mp = layers.GlobalAveragePooling2D() (conv)
        mp = layers.Dense(nb_chan // SE_prob, activation = 'relu') (mp)
        mp = layers.Dense(nb_chan, activation = 'sigmoid') (mp)
        conv = layers.Multiply () ([conv,mp])

    conv = PConv(filters = n_filters,
                    mask = mask,
                    over_compensation = over_compensation,
                    kernel_size = kernel_size,
                    padding = 'same',
                    kernel_initializer = kernel_initializer,
                    activation = activation)(conv)
    if batch_norm == True : 
        conv = layers.BatchNormalization()(conv)
    conv = layers.MaxPooling2D(pool_size=(2, 2))(conv)
    mask = layers.MaxPooling2D(pool_size=(2, 2))(mask)

    return conv, mask

def bottleneck( prev_layer, mask, n_filters, kernel_size,over_compensation=True, batch_norm = True, activation = 'relu', kernel_initializer='HeNormal',SE_prob = 0, dropout_prob = 0.2) : 

    conv = PConv(filters = n_filters,
                    mask = mask,
                    over_compensation = over_compensation,
                    kernel_size = kernel_size,
                    padding = 'same',
                    kernel_initializer = kernel_initializer,
                    activation = activation)(prev_layer)

    if dropout_prob > 0:
        conv = layers.Dropout(dropout_prob)(conv)

    conv = PConv(filters = n_filters,
                    mask = mask,
                    kernel_size = kernel_size,
                    over_compensation = over_compensation,
                    padding = 'same',
                    kernel_initializer = kernel_initializer,
                    activation = activation)(conv)
    if SE_prob >0 :
        nb_chan = keras.backend.int_shape(conv)[-1]
        mp = layers.GlobalAveragePooling2D() (conv)
        mp = layers.Dense(nb_chan // SE_prob, activation = 'relu') (mp)
        mp = layers.Dense(nb_chan, activation = 'sigmoid') (mp)
        conv = layers.Multiply () ([conv,mp])
    conv = PConv(filters = n_filters,
                    mask = mask,
                    kernel_size = kernel_size,
                    over_compensation = over_compensation,
                    padding = 'same',
                    kernel_initializer = kernel_initializer,
                    activation = activation)(conv)
    if SE_prob >0 :
        nb_chan = keras.backend.int_shape(conv)[-1]
        mp = layers.GlobalAveragePooling2D() (conv)
        mp = layers.Dense(nb_chan // SE_prob, activation = 'relu') (mp)
        mp = layers.Dense(nb_chan, activation = 'sigmoid') (mp)
        conv = layers.Multiply () ([conv,mp])
    conv = PConv(filters = n_filters,
                    mask = mask,
                    kernel_size =  kernel_size,
                    over_compensation = over_compensation,
                    padding = 'same',
                    kernel_initializer = kernel_initializer,
                    activation = activation)(conv)

    if batch_norm == True :
        conv = layers.BatchNormalization()(conv)

    return conv, mask


def decoder_block(prev_layer, skip_layer, mask, n_filters,over_compensation = True,kernel_size = 3, batch_norm = True, SE_prob=0, activation = 'relu', kernel_initializer='HeNormal'):

    up = PConv(filters = n_filters,
                    mask = mask,
                    over_compensation = over_compensation,
                    kernel_size = kernel_size,
                    activation = activation,
                    kernel_initializer=kernel_initializer,
                    padding = 'same')(layers.UpSampling2D(size = (2,2))(prev_layer))
    merge = layers.concatenate([up, skip_layer], axis = 3)
    conv = PConv(filters = n_filters,
                    mask=mask,
                    over_compensation = over_compensation,
                    kernel_size = kernel_size,
                    activation = activation,
                    kernel_initializer=kernel_initializer,
                    padding = 'same')(merge)
    if SE_prob >0 :
        nb_chan = keras.backend.int_shape(conv)[-1]
        mp = layers.GlobalAveragePooling2D() (conv)
        mp = layers.Dense(nb_chan // SE_prob, activation = 'relu') (mp)
        mp = layers.Dense(nb_chan, activation = 'sigmoid') (mp)
        conv = layers.Multiply () ([conv,mp])
    conv = PConv(filters = n_filters,
                    mask=mask,
                    kernel_size = kernel_size,
                    over_compensation = over_compensation,
                    activation = activation,
                    kernel_initializer=kernel_initializer,
                    padding = 'same')(conv)

    conv = PConv(filters = n_filters,
                    mask=mask,
                    over_compensation = over_compensation,
                    kernel_size = kernel_size,
                    activation = activation,
                    kernel_initializer=kernel_initializer,
                    padding = 'same')(conv)

    if batch_norm == True :
        conv = layers.BatchNormalization()(conv)

    return conv

class UNet5(tf.keras.Model):
    '''Build a UNet with any skip connections
        Input : - mask : initial mask the partial data
                - kernel_size : int, size of the kernel for the partial convolution, except the last one
                - activation : string, activation function, as known by keras for Conv2D
                - filters : array of 4 int [32, 64,128,256] for example
                    which give the size of the filters
                    for the several convolution after each reduction phase'''

    def __init__(self,mask,input_shape = (288, 256, 10), kernel_size=3,activation='relu',depth = 3, n_filter = 4,SE_prob= 0,  dropout_prob=0.2):
        super(UNet5,self).__init__()

        mask = tf.expand_dims(mask, axis = 0)

        inputs_img = layers.Input(input_shape, name = 'inputs_img')
        kernel_initializer = tf.keras.initializers.HeNormal()
        
        conv = PConv(filters = n_filter,
                    mask=mask,
                    kernel_size = 3,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(inputs_img)

        conv1, mask1 = encoder_block (conv, 
                                        mask = mask, 
                                        n_filters = n_filter, 
                                        activation = activation, 
                                        kernel_initializer=kernel_initializer, 
                                        dropout_prob = dropout_prob, 
                                        SE_prob=SE_prob,
                                        batch_norm = True)
        
        conv2, mask2 = encoder_block (conv1,
                                        mask = mask1,
                                        n_filters = 2*n_filter,
                                        activation = activation,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob,
                                        SE_prob=SE_prob,
                                        batch_norm = True)

        conv3, mask3 = encoder_block (conv2,
                                        mask = mask2,
                                        n_filters = 4*n_filter,
                                        activation = activation,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob,
                                        SE_prob=SE_prob,
                                        batch_norm = True)

        conv4, mask4 = encoder_block (conv3,
                                        mask = mask3,
                                        n_filters = 8*n_filter,
                                        activation = activation,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob,
                                        SE_prob=SE_prob,
                                        batch_norm = True)

        convbn, maskbn = bottleneck( conv4, 
                                        mask4, 
                                        n_filters = 16*n_filter, 
                                        kernel_size= 3, 
                                        batch_norm = True, 
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer, 
                                        dropout_prob = dropout_prob)

        conv5 = decoder_block (convbn,conv3,
                                        mask = mask3,
                                        n_filters = 8*n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        batch_norm = True)

        conv6 = decoder_block (conv5,conv2,
                                        mask = mask2,
                                        n_filters = 4*n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        batch_norm = True)
        conv7 = decoder_block (conv6, conv1,
                                        mask = mask1,
                                        n_filters = 2*n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        batch_norm = True)
        conv8 = decoder_block (conv7, conv,
                                        mask = mask,
                                        n_filters = n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        batch_norm = True)


        outputs1 = PConv(filters = 2,
                    mask=mask,
                    kernel_size = 1,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = tf.keras.initializers.Zeros())(conv8)

        self.model = tf.keras.Model(inputs = inputs_img, outputs = outputs1)
        self.model.summary()
    def call(self, inputs_img):
        return self.model(inputs_img)


class UNet4(tf.keras.Model):
    '''Build a UNet with any skip connections
        Input : - mask : initial mask the partial data
                - kernel_size : int, size of the kernel for the partial convolution, except the last one
                - activation : string, activation function, as known by keras for Conv2D
                - filters : array of 4 int [32, 64,128,256] for example
                    which give the size of the filters
                    for the several convolution after each reduction phase'''

    def __init__(self,mask,input_shape = (288, 256, 10), kernel_size=3,activation='relu',SE_prob = 0.3,depth = 3, n_filter = 4, dropout_prob=0.2):
        super(UNet4,self).__init__()

        mask = tf.expand_dims(mask, axis = 0)
        print(np.shape(mask))
        inputs_img = layers.Input(input_shape, name = 'inputs_img')
        kernel_initializer = tf.keras.initializers.HeNormal()

        conv = PConv(filters = n_filter,
                    mask=mask,
                    kernel_size = 3,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(inputs_img)

        conv1, mask1 = encoder_block (conv,
                                        mask = mask,
                                        n_filters = n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob,
                                        batch_norm = True)

        conv2, mask2 = encoder_block (conv1,
                                        mask = mask1,
                                        n_filters = 2*n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob,
                                        batch_norm = True)

        conv3, mask3 = encoder_block (conv2,
                                        mask = mask2,
                                        n_filters = 4*n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob,
                                        batch_norm = True)


        convbn, maskbn = bottleneck( conv3,
                                        mask3,
                                        n_filters = 8*n_filter,
                                        kernel_size= 3,
                                        batch_norm = True,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob)


        conv6 = decoder_block (convbn,conv2,
                                        mask = mask2,
                                        n_filters = 4*n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        batch_norm = True)
        conv7 = decoder_block (conv6, conv1,
                                        mask = mask1,
                                        n_filters = 2*n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        batch_norm = True)
        conv8 = decoder_block (conv7, conv,
                                        mask = mask,
                                        n_filters = n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        batch_norm = True)


        outputs1 = PConv(filters = 2,
                    mask=mask,
                    kernel_size = 1,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = tf.keras.initializers.Zeros())(conv8)

        self.model = tf.keras.Model(inputs = inputs_img, outputs = outputs1)
        self.model.summary()
    def call(self, inputs_img):
        return self.model(inputs_img)


class UNet3(tf.keras.Model):
    '''Build a UNet with any skip connections
        Input : - mask : initial mask the partial data
                - kernel_size : int, size of the kernel for the partial convolution, except the last one
                - activation : string, activation function, as known by keras for Conv2D
                - filters : array of 4 int [32, 64,128,256] for example
                    which give the size of the filters
                    for the several convolution after each reduction phase'''

    def __init__(self,mask,input_shape = (288, 256, 10),N_output=2,over_compensation = True, kernel_size=3,SE_prob = 0.3,activation='relu',depth = 3, n_filter = 4, dropout_prob=0.2):
        super(UNet3,self).__init__()

        mask = tf.expand_dims(mask, axis = 0)

        inputs_img = layers.Input(input_shape, name = 'inputs_img')
        kernel_initializer = tf.keras.initializers.HeNormal()
        
        conv = PConv(filters = n_filter,
                    mask=mask,
                    over_compensation = over_compensation,
                    kernel_size = kernel_size,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(inputs_img)

        conv1, mask1 = encoder_block (conv,
                                        mask = mask,
                                        n_filters = n_filter,
                                        over_compensation = over_compensation,
                                        kernel_size = kernel_size,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob,
                                        batch_norm = True)

        conv2, mask2 = encoder_block (conv1,
                                        mask = mask1,
                                        n_filters = 2*n_filter,
                                        over_compensation = over_compensation,
                                        kernel_size = kernel_size,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob,
                                        batch_norm = True)


        convbn, maskbn = bottleneck( conv2,
                                        mask2,
                                        n_filters = 8*n_filter,
                                        over_compensation = over_compensation,
                                        kernel_size = kernel_size,
                                        batch_norm = True,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob)


        conv7 = decoder_block (convbn, conv1,
                                        mask = mask1,
                                        over_compensation = over_compensation,
                                        kernel_size = kernel_size,
                                        n_filters = 2*n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        batch_norm = True)
        conv8 = decoder_block (conv7, conv,
                                        mask = mask,
                                        over_compensation = over_compensation,
                                        kernel_size = kernel_size,
                                        n_filters = n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        batch_norm = True)


        outputs1 = PConv(filters = N_output,
                    mask=mask,
                    over_compensation = over_compensation,
                    kernel_size = 1,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = tf.keras.initializers.Zeros())(conv8)

        self.model = tf.keras.Model(inputs = inputs_img, outputs = outputs1)
        self.model.summary()
    def call(self, inputs_img):
        return self.model(inputs_img)


class UNet2(tf.keras.Model):
    '''Build a UNet with any skip connections
        Input : - mask : initial mask the partial data
                - kernel_size : int, size of the kernel for the partial convolution, except the last one
                - activation : string, activation function, as known by keras for Conv2D
                - filters : array of 4 int [32, 64,128,256] for example
                    which give the size of the filters
                    for the several convolution after each reduction phase'''

    def __init__(self,mask,input_shape = (288, 256, 10),SE_prob = 0.3, kernel_size=3,activation='relu',depth = 3, n_filter = 4, dropout_prob=0.2):
        super(UNet2,self).__init__()

        mask = tf.expand_dims(mask, axis = 0)

        inputs_img = layers.Input(input_shape, name = 'inputs_img')
        kernel_initializer = tf.keras.initializers.HeNormal()

        conv = PConv(filters = n_filter,
                    mask=mask,
                    kernel_size = 3,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(inputs_img)

        conv1, mask1 = encoder_block (conv,
                                        mask = mask,
                                        n_filters = n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob,
                                        batch_norm = True)



        convbn, maskbn = bottleneck( conv1,
                                        mask1,
                                        n_filters = 8*n_filter,
                                        kernel_size= 3,
                                        batch_norm = True,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob)


        conv8 = decoder_block (convbn, conv,
                                        mask = mask,
                                        n_filters = n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        batch_norm = True)


        outputs1 = PConv(filters = 2,
                    mask=mask,
                    kernel_size = 1,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = tf.keras.initializers.Zeros())(conv8)

        self.model = tf.keras.Model(inputs = inputs_img, outputs = outputs1)
        self.model.summary()
    def call(self, inputs_img):
        return self.model(inputs_img)





class UNet4_1output(tf.keras.Model):
    '''Build a UNet with any skip connections
        Input : - mask : initial mask the partial data
                - kernel_size : int, size of the kernel for the partial convolution, except the last one
                - activation : string, activation function, as known by keras for Conv2D
                - filters : array of 4 int [32, 64,128,256] for example
                    which give the size of the filters
                    for the several convolution after each reduction phase'''

    def __init__(self,mask,input_shape = (288, 256, 10), kernel_size=3,activation='relu',SE_prob = 0.3,depth = 3, n_filter = 4, dropout_prob=0.2):
        super(UNet4_1output,self).__init__()

        mask = tf.expand_dims(mask, axis = 0)

        inputs_img = layers.Input(input_shape, name = 'inputs_img')
        kernel_initializer = tf.keras.initializers.HeNormal()

        conv = PConv(filters = n_filter,
                    mask=mask,
                    kernel_size = 3,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(inputs_img)

        conv1, mask1 = encoder_block (conv,
                                        mask = mask,
                                        n_filters = n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob,
                                        batch_norm = True)

        conv2, mask2 = encoder_block (conv1,
                                        mask = mask1,
                                        n_filters = 2*n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob,
                                        batch_norm = True)

        conv3, mask3 = encoder_block (conv2,
                                        mask = mask2,
                                        n_filters = 4*n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob,
                                        batch_norm = True)


        convbn, maskbn = bottleneck( conv3,
                                        mask3,
                                        n_filters = 8*n_filter,
                                        kernel_size= 3,
                                        batch_norm = True,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob)


        conv6 = decoder_block (convbn,conv2,
                                        mask = mask2,
                                        n_filters = 4*n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        batch_norm = True)
        conv7 = decoder_block (conv6, conv1,
                                        mask = mask1,
                                        n_filters = 2*n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        batch_norm = True)
        conv8 = decoder_block (conv7, conv,
                                        mask = mask,
                                        n_filters = n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        batch_norm = True)


        outputs1 = PConv(filters = 1,
                    mask=mask,
                    kernel_size = 1,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = tf.keras.initializers.Zeros())(conv8)

        self.model = tf.keras.Model(inputs = inputs_img, outputs = outputs1)
        self.model.summary()
    def call(self, inputs_img):
        return self.model(inputs_img)

class CNN(tf.keras.Model):
    '''Build a CNN with any skip connections
        Input : - mask : initial mask the partial data
                - kernel_size : int, size of the kernel for the partial convolution, except the last one
                - activation : string, activation function, as known by keras for Conv2D
                - filters : array of 4 int [32, 64,128,256] for example
                    which give the size of the filters
                    for the several convolution after each reduction phase'''

    def __init__(self,mask,N_output = 2, input_shape = (288, 256, 10), kernel_size=3,SE_prob = 0.3,activation='relu',depth = 3, n_filter = 4, dropout_prob=0.2):
        super(CNN,self).__init__()

        

        inputs_img = layers.Input(input_shape, name = 'inputs_img')
        kernel_initializer = tf.keras.initializers.HeNormal()

        conv = layers.Conv2D(filters = n_filter,
                    kernel_size = 3,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(inputs_img)
        conv = layers.Conv2D(filters = 2*n_filter,
                    kernel_size = 3,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(conv)
        conv = layers.Conv2D(filters = 4*n_filter,
                    kernel_size = 3,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(conv)
        conv = layers.Conv2D(filters = 8*n_filter,
                    kernel_size = 3,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(conv)
        conv = layers.Conv2D(filters = 16*n_filter,
                    kernel_size = 3,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(conv)
        conv = layers.Conv2D(filters = 8*n_filter,
                    kernel_size = 3,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(conv)
        conv = layers.Conv2D(filters = 4*n_filter,
                    kernel_size = 3,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(conv)
        conv = layers.Conv2D(filters = 2*n_filter,
                    kernel_size = 3,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(conv)

        outputs1 = layers.Conv2D(filters = N_output,
                    kernel_size = 1,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = tf.keras.initializers.Zeros())(conv)

        self.model = tf.keras.Model(inputs = inputs_img, outputs = outputs1)
        self.model.summary()
    def call(self, inputs_img):
        return self.model(inputs_img)
class PCNN(tf.keras.Model):
    '''Build a CNN with any skip connections
        Input : - mask : initial mask the partial data
                - kernel_size : int, size of the kernel for the partial convolution, except the last one
                - activation : string, activation function, as known by keras for Conv2D
                - filters : array of 4 int [32, 64,128,256] for example
                    which give the size of the filters
                    for the several convolution after each reduction phase'''

    def __init__(self,mask,N_output = 2,input_shape = (288, 256, 10), kernel_size=3,SE_prob = 0.3,activation='relu',depth = 3, n_filter = 4, dropout_prob=0.2):
        super(PCNN,self).__init__()


        mask = tf.expand_dims(mask, axis = 0)
        inputs_img = layers.Input(input_shape, name = 'inputs_img')
        kernel_initializer = tf.keras.initializers.HeNormal()

        conv = PConv(filters = n_filter,
                    mask = mask,
                    kernel_size = 3,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(inputs_img)
        conv = PConv(filters = 2*n_filter,
                    mask = mask,
                    kernel_size = 3,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(conv)
        conv = PConv(filters = 4*n_filter,
                    mask = mask,
                    kernel_size = 3,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(conv)
        conv = PConv(filters = 8*n_filter,
                    kernel_size = 3,
                    mask = mask,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(conv)
        conv = PConv(filters = 16*n_filter,
                    mask = mask,
                    kernel_size = 3,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(conv)
        conv = PConv(filters = 8*n_filter,
                    kernel_size = 3,
                    mask = mask,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(conv)
        conv = PConv(filters = 4*n_filter,
                    kernel_size = 3,
                    mask = mask,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(conv)
        conv = PConv(filters = 2*n_filter,
                    kernel_size = 3,
                    mask = mask,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(conv)

        outputs1 = PConv(filters = N_output,
                    kernel_size = 1,
                    mask = mask,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = tf.keras.initializers.Zeros())(conv)

        self.model = tf.keras.Model(inputs = inputs_img, outputs = outputs1)
        self.model.summary()
    def call(self, inputs_img):
        return self.model(inputs_img)



class ResNet(tf.keras.Model):
    ''' Build a ResNet Model '''
    def __init__(self, mask,input_shape,n_filter,N_output=2, dropout_prob=0.2, SE_prob = 0,depth = 0, kernel_size = 5, activation = 'relu',filters=[16,32,64],reg=1e-6,dropout=0.3):
        super(ResNet,self).__init__()
        F1,F2,F3 = filters
        mask = tf.expand_dims(mask, axis = 0)
        X_input = layers.Input(input_shape, name = 'inputs_img')

        X = PConv(filters = F3,mask=mask,kernel_size = 3,padding = 'same',
                            kernel_initializer = tf.keras.initializers.HeNormal())(X_input)

        X_shortcut = X
        X = layers.BatchNormalization()(X)
        X = PConv(filters = F1,
                            mask = mask,
                            kernel_size = kernel_size,
                            padding = 'same',
                            activation = activation,
                            bias_initializer="zeros",
                            kernel_initializer = tf.keras.initializers.HeNormal())(X)
        X = PConv(filters = F2,
                            mask =mask,
                            kernel_size = kernel_size,
                            padding = 'same',
                            activation = activation,
                            bias_initializer="zeros",
                            kernel_initializer = tf.keras.initializers.HeNormal())(X)
        X = PConv(filters = F3,
                            mask = mask,
                            kernel_size = kernel_size,
                            padding = 'same',
                            activation = activation,
                            kernel_initializer='zeros')(X)

        X = X + X_shortcut
        X_shortcut = X
        X = layers.BatchNormalization()(X)
        X = PConv(filters = F1,
                            mask = mask,
                            kernel_size = kernel_size,
                            padding = 'same',
                            bias_initializer="zeros",
                            activation = activation,
                            kernel_initializer = tf.keras.initializers.HeNormal())(X)
        X = PConv(filters = F2,
                            mask = mask,
                            kernel_size = kernel_size,
                            padding = 'same',
                            bias_initializer="zeros",
                            activation = activation,
                            kernel_initializer = tf.keras.initializers.HeNormal())(X)
        X = PConv(filters = F3,
                            mask = mask,
                            kernel_size = kernel_size,
                            padding = 'same',
                            activation = activation,
                            kernel_initializer='zeros')(X)

        X = X + X_shortcut
        X_shortcut = X
        X = layers.BatchNormalization()(X)
        X = PConv(filters = F1,
                            mask = mask,
                            kernel_size = kernel_size,
                            padding = 'same',
                            activation = activation,
                            kernel_initializer = tf.keras.initializers.HeNormal(),
                            bias_initializer="zeros")(X)
        X = PConv(filters = F2,
                            mask = mask,
                            kernel_size = kernel_size,
                            padding = 'same',
                            activation = activation,
                            kernel_initializer = tf.keras.initializers.HeNormal(),
                            bias_initializer="zeros")(X)
        X = PConv(filters = F3,
                            mask = mask,
                            kernel_size = kernel_size,
                            padding = 'same',
                            activation = activation,
                            kernel_initializer='zeros')(X)
       # X_shortcut = PConv2D(filters = F3,
        #                    mask = mask,
         #                   kernel_size = kernel_size,
          #                  padding = 'same',
           #                 kernel_initializer = tf.keras.initializers.HeNormal(),
            #                bias_initializer="zeros")(X_shortcut)

        X = X + X_shortcut

        X_shortcut = X
        X = layers.BatchNormalization()(X)
        X = PConv(filters = F1,
                            mask = mask,
                            kernel_size = kernel_size,
                            padding = 'same',
                            activation = activation,
                            kernel_initializer = tf.keras.initializers.HeNormal(),
                            bias_initializer="zeros")(X)
        X = PConv(filters = F2,
                            mask = mask,
                            kernel_size = kernel_size,
                            padding = 'same',
                            activation = activation,
                            kernel_initializer = tf.keras.initializers.HeNormal(),
                            bias_initializer="zeros")(X)
        X = PConv(filters = F3,
                            mask = mask,
                            kernel_size = kernel_size,
                            padding = 'same',
                            activation = activation,
                            kernel_initializer='zeros')(X)

        X = X + X_shortcut

        X_shortcut = X
        X = layers.BatchNormalization()(X)
        X = PConv(filters = F1,
                            mask = mask,
                            kernel_size = kernel_size,
                            padding = 'same',
                            activation = activation,
                            kernel_initializer = tf.keras.initializers.HeNormal(),
                            bias_initializer="zeros")(X)
        X = PConv(filters = F2,
                            mask = mask,
                            kernel_size = kernel_size,
                            padding = 'same',
                            activation = activation,
                            kernel_initializer = tf.keras.initializers.HeNormal(),
                            bias_initializer="zeros")(X)
        X = PConv(filters = F3,
                            mask = mask,
                            kernel_size = kernel_size,
                            padding = 'same',
                            activation = activation,
                            kernel_initializer='zeros')(X)
        #X_shortcut = PConv2D(filters = F3,
         #                   mask = mask,
          #                  kernel_size = kernel_size,
           #                 padding = 'same',
        #                    kernel_initializer = tf.keras.initializers.HeNormal(),
         #                   bias_initializer="zeros")(X_shortcut)

        X = X + X_shortcut
        X = layers.BatchNormalization()(X)
        X = PConv(filters = 8,kernel_size = 3,mask = mask,
                            padding = 'same',
                            kernel_initializer = tf.keras.initializers.HeNormal())(X)
        outputs = PConv(filters = N_output,kernel_size = 1,mask = mask, padding = 'same', activation = 'linear',kernel_initializer='zeros')(X)

        self.model = tf.keras.Model(inputs = X_input, outputs = outputs)
        self.model.summary()
    def call(self, inputs_img):
        return self.model(inputs_img)





class UNet3_1output(tf.keras.Model):
    '''Build a UNet with any skip connections
        Input : - mask : initial mask the partial data
                - kernel_size : int, size of the kernel for the partial convolution, except the last one
                - activation : string, activation function, as known by keras for Conv2D
                - filters : array of 4 int [32, 64,128,256] for example
                    which give the size of the filters
                    for the several convolution after each reduction phase'''

    def __init__(self,mask,input_shape = (288, 256, 10), kernel_size=3,SE_prob = 0.3,activation='relu',depth = 3, n_filter = 4, dropout_prob=0.2):
        super(UNet3_1output,self).__init__()

        mask = tf.expand_dims(mask, axis = 0)

        inputs_img = layers.Input(input_shape, name = 'inputs_img')
        kernel_initializer = tf.keras.initializers.HeNormal()

        conv = PConv(filters = n_filter,
                    mask=mask,
                    kernel_size = 3,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(inputs_img)

        conv1, mask1 = encoder_block (conv,
                                        mask = mask,
                                        n_filters = n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob,
                                        batch_norm = True)

        conv2, mask2 = encoder_block (conv1,
                                        mask = mask1,
                                        n_filters = 2*n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob,
                                        batch_norm = True)


        convbn, maskbn = bottleneck( conv2,
                                        mask2,
                                        n_filters = 8*n_filter,
                                        kernel_size= 3,
                                        batch_norm = True,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob)


        conv7 = decoder_block (convbn, conv1,
                                        mask = mask1,
                                        n_filters = 2*n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        batch_norm = True)
        conv8 = decoder_block (conv7, conv,
                                        mask = mask,
                                        n_filters = n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        batch_norm = True)


        outputs1 = PConv(filters = 1,
                    mask=mask,
                    kernel_size = 1,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = tf.keras.initializers.Zeros())(conv8)

        self.model = tf.keras.Model(inputs = inputs_img, outputs = outputs1)
        self.model.summary()
    def call(self, inputs_img):
        return self.model(inputs_img)

class UNet3_withsum(tf.keras.Model):
    '''Build a UNet with any skip connections
        Input : - mask : initial mask the partial data
                - kernel_size : int, size of the kernel for the partial convolution, except the last one
                - activation : string, activation function, as known by keras for Conv2D
                - filters : array of 4 int [32, 64,128,256] for example
                    which give the size of the filters
                    for the several convolution after each reduction phase'''

    def __init__(self,mask,input_shape = (288, 256, 10), kernel_size=3,SE_prob = 0.3,activation='relu',depth = 3, n_filter = 4, dropout_prob=0.2):
        super(UNet3_withsum,self).__init__()

        mask = tf.expand_dims(mask, axis = 0)

        inputs_img = layers.Input(input_shape, name = 'inputs_img')
        kernel_initializer = tf.keras.initializers.HeNormal()
        
        sea_ice_sum = tf.math.reduce_mean(inputs_img[:,:,:,0])
        conv = PConv(filters = n_filter,
                    mask=mask,
                    kernel_size = 3,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = kernel_initializer)(inputs_img)

        conv1, mask1 = encoder_block (conv,
                                        mask = mask,
                                        n_filters = n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob,
                                        batch_norm = True)

        conv2, mask2 = encoder_block (conv1,
                                        mask = mask1,
                                        n_filters = 2*n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob,
                                        batch_norm = True)


        convbn, maskbn = bottleneck( conv2,
                                        mask2,
                                        n_filters = 8*n_filter,
                                        kernel_size= 3,
                                        batch_norm = True,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        dropout_prob = dropout_prob)


        conv7 = decoder_block (convbn, conv1,
                                        mask = mask1,
                                        n_filters = 2*n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        batch_norm = True)
        conv8 = decoder_block (conv7, conv,
                                        mask = mask,
                                        n_filters = n_filter,
                                        activation = activation,
                                        SE_prob=SE_prob,
                                        kernel_initializer=kernel_initializer,
                                        batch_norm = True)


        outputs1 = PConv(filters = 2,
                    mask=mask,
                    kernel_size = 1,
                    padding="same",
                    use_bias=True,
                    kernel_initializer = tf.keras.initializers.Zeros())(conv8)
        ii = tf.ones((256, 128,128,1), dtype = tf.float32)
        print(ii)
        outputs_sum = tf.math.reduce_mean(outputs1[:,:,:,0])
        outputs_sum = tf.math.multiply(ii, outputs_sum)
        print(outputs_sum)
        self.model = tf.keras.Model(inputs = inputs_img, outputs = tf.concat([outputs1, outputs_sum], axis = 3))
        self.model.summary()
    def call(self, inputs_img):
        return self.model(inputs_img)
