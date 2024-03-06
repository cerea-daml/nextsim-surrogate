import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import tensorflow as tf
import xarray as xr
from tqdm import trange
import matplotlib.pyplot as plt
from functools import partial

from skimage.measure import block_reduce
import tensorflow as tf
from tensorflow import keras



def replacenan(t):
    '''
    replace nan value in a tensorflow tensor
    Parameters :
    -------------------
    t : tensorflow array

    Outputs :
    --------------------
    tensor array of the same shape
    '''
    return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)


def correlation(u, v):
    '''
    Compute spatial correlation of two array of same dimensions
    Parameters :
    --------------------
    u, v : two numpy arrays of same dimensions

    Outputs :
    --------------------
    Scalar between 0 and 1
    '''

    term1 = u - u.mean()
    term2 = v - v.mean()

    corr = (
        np.shape(term1)[0]
        * (term1 * term2).mean()
        / (np.linalg.norm(term1) * np.linalg.norm(term2))
    )

    return corr


class Test:
    '''
    Python class to test a surrogate model
    '''
    def __init__(self, model, mask, k, N_cycle,  timestep, save_pred, path_to_save, path_to_data, noise_init, noise):
        '''
	Parameters : 
        -------------------
        model : Tensorflow model initialized with its subclass
        mask : 2D array with 0 for land pixel and 1 for sea/ice
        time : 1D np.array containing time of snapshots
        timestep : int, number of inputs in the dataset
        path_to_save : str, path where the weights of the NN are 
        path_to_data :str, path to tfrecords files
        N_cycle: number of initialization: for test year in 2018, select 1370, 
                    for seasonal forecast in 2018, 1, 
                    and seasonal forecast in 2006-2008, 25
        save_pred: bool, whether the prediction are saved, True, if you need to compute PSD/analyse visually the trajectory (important disk place)
        '''
        self.model = model
        self.mask = mask
        self.path = path_to_save
        self.timestep = timestep
        self.N_cycle = N_cycle
        self.timestep_output = 2
        self.path_data = path_to_data
        self.noise_init = noise_init
        self.noise = noise
        self.save_pred = save_pred
        
        #Normalization constant
        #self.mean_input = 0.38418264491680904
        self.mean_input =0.38415005911946803
        #self.std_input = 0.771096968092377
        self.std_input = 0.8016633217452184
        #self.mean_output = -1.6270055141928728e-05
        self.mean_output = -3.541132e-05
        #self.std_output = 0.023030024601018818
        self.std_output = 0.042275164
        self.N_x = 256
        self.N_y = 256

        self.k = k
        

    def normalize_input(self, x):
        return (x - self.mean_input) / self.std_input

    def reverse_normalize_output(self, x):
        return  x * self.std_output + self.mean_output 

    def reverse_normalize_input(self, x):
        return  x * self.std_input + self.mean_input 

    def get_dataset(self, filenames, batch_size):
        dataset = self.load_dataset(filenames, self.timestep)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.repeat()
        return dataset

    def load_dataset(self, filenames, labeled=True):
        dataset = tf.data.TFRecordDataset(
            filenames
        )  # automatically interleaves reads from multiple files
        dataset.element_spec
        if self.timestep == 2:
            dataset = dataset.map(partial(self.read_tfrecord2))
        elif self.timestep == 1:
            dataset = dataset.map(partial(self.read_tfrecord1))
        return dataset

    def read_tfrecord1(self, example):

        tfrecord_format = {
            "inputs": tf.io.FixedLenFeature(
                [4 * self.N_x * self.N_y * 4], tf.float32
            ),
            "outputs": tf.io.FixedLenFeature(
                [self.timestep_output * self.N_x * self.N_y], tf.float32
            ),
        }

        example = tf.io.parse_single_example(example, tfrecord_format)

        inputs = tf.cast(example["inputs"], tf.float32)

        inputs = tf.reshape(inputs, [*[4,  self.N_x,  self.N_x, 4]])
        inputs = tf.transpose(inputs, [1, 2, 0, 3])
        inputs_forcings = inputs[:,:,1:,-2:]
        inputs_forcings = tf.reshape(inputs_forcings, [*[ self.N_x,  self.N_x, 6]])
        inputs_past = inputs[:,:,:,1]
        inputs_past = tf.reshape(inputs_past, [*[ self.N_x,  self.N_x, 4]])
        inputs = tf.concat([inputs_past, inputs_forcings], axis = 2)
        outputs = tf.cast(example["outputs"], tf.float32)
        inputs = replacenan(inputs)
        outputs = tf.reshape(outputs, [*[ self.N_x,  self.N_x, 2]])
        return inputs, outputs

    def read_tfrecord2(self, example):

        tfrecord_format = {
            "inputs": tf.io.FixedLenFeature(
                [4 * self.N_x * self.N_y * 4], tf.float32
            ),
            "outputs": tf.io.FixedLenFeature(
                [self.timestep_output * self.N_x * self.N_y], tf.float32
            ),
        }

        example = tf.io.parse_single_example(example, tfrecord_format)
        inputs = tf.cast(example["inputs"], tf.float32)

        inputs = tf.reshape(inputs, [*[4, self.N_x, self.N_y, 4]])
        inputs = tf.transpose(inputs, [1, 2, 0, 3])
        inputs_forcings = inputs[:,:,1:,-2:]
        inputs_forcings = tf.reshape(inputs_forcings, [*[self.N_x, self.N_y, 6]])
        inputs_past = inputs[:,:,:,:2]
        inputs_past = tf.reshape(inputs_past, [*[self.N_x, self.N_y, 4 * 2]])
        inputs = tf.concat([inputs_past, inputs_forcings], axis = 2)
        outputs = tf.cast(example["outputs"], tf.float32)
        inputs = replacenan(inputs)
        outputs = tf.reshape(outputs, [*[self.N_x, self.N_y, 2]])


        return inputs, outputs

    def add_noise(self, x):
        return x + np.random.normal(0, self.noise, (self.N_x, self.N_y, self.timestep))
    def FS(self, x_truth):
        x_truth = x_truth.numpy()
        min_truth = np.min(x_truth[0,:,:,0])
        
        result = np.zeros((self.k))
        x_inputs = np.zeros((self.k, self.N_x, self.N_y, 4 * self.timestep + 6))
        x_pred = np.zeros((self.k, self.N_x, self.N_y, 1))   
        x_truth[0, :, :, :self.timestep]   = self.add_noise(x_truth[0, :, :, :self.timestep]) 
        x_inputs[0] = x_truth[0].reshape((self.N_x, self.N_y, 4 * self.timestep + 6))
        x_pred[0] = x_truth[0, :, :, self.timestep - 1].reshape((self.N_x, self.N_y, 1))
    
        for t in range(1, self.k):

            err = self.model.predict(
                x_inputs[t - 1].reshape((1, self.N_x, self.N_y, 4 * self.timestep + 6)), verbose = 0
            )
            err = err[0, :, :, 0].reshape((self.N_x, self.N_y, 1))
            err = self.reverse_normalize_output(err)
            x_pred[t] = self.reverse_normalize_input(x_pred[t - 1]) + err
            x_pred[t] = self.normalize_input(x_pred[t]) 
            x_pred[t] = np.clip(x_pred[t], a_min = min_truth, a_max = None)
            
            if self.timestep != 1 :
            	x_inputs[t] = np.concatenate(
                    (
                    x_inputs[t - 1, :, :, 1:self.timestep].reshape((self.N_x, self.N_y, self.timestep - 1)),
                    x_pred[t].reshape((self.N_x, self.N_y, 1)),
                    x_truth[2 * t, :, :, self.timestep :],
                    ),
                    axis=2,
                )
            else :
                x_inputs[t] = np.concatenate(
                    (
                    x_pred[t].reshape((self.N_x, self.N_y, 1)),
                    x_truth[2 * t, :, :, self.timestep :],
                    ),
                    axis=2,
                )
        bias = ( x_pred.squeeze()* self.mask.squeeze() - x_truth[0 : 2 * self.k : 2, :, :, self.timestep - 1]* self.mask.squeeze()).mean(axis = (1,2))
        result = np.sqrt(
            (
                (
                    x_pred.squeeze()* self.mask.squeeze()
                    - x_truth[0 : 2 * self.k : 2, :, :, self.timestep - 1]* self.mask.squeeze()
                )
                ** 2
            ).mean(axis=(1, 2))
        )
        result_pers = np.sqrt(
            (
                (
                    x_truth[0, :, :, self.timestep - 1]* self.mask.squeeze()
                    - x_truth[0 : 2 * self.k : 2, :, :, self.timestep - 1]* self.mask.squeeze()
                )
                ** 2
            ).mean(axis=(1, 2))
        )

        return result, result_pers, bias, x_pred

    def test_model(self):

        TEST_FILENAMES = [
                self.path_data + "test.tfrecords.000",
                self.path_data + "test.tfrecords.001",
                self.path_data + "test.tfrecords.002",
                self.path_data + "test.tfrecords.003",
                self.path_data + "test.tfrecords.004",
                self.path_data + "test.tfrecords.005",
                self.path_data + "test.tfrecords.006",
                self.path_data + "test.tfrecords.007",
                self.path_data + "test.tfrecords.008",
                self.path_data + "test.tfrecords.009",
                self.path_data + "test.tfrecords.010",
                self.path_data + "test.tfrecords.011",
                self.path_data + "test.tfrecords.012",
                self.path_data + "test.tfrecords.013",
                self.path_data + "test.tfrecords.014",
                self.path_data + "test.tfrecords.015",
                self.path_data + "test.tfrecords.016",
                self.path_data + "test.tfrecords.017",
                self.path_data + "test.tfrecords.018",
                self.path_data + "test.tfrecords.019",
                self.path_data + "test.tfrecords.020",
                self.path_data + "test.tfrecords.021",
                self.path_data + "test.tfrecords.022",
                self.path_data + "test.tfrecords.023",
                self.path_data + "test.tfrecords.024"]
        fs = np.zeros((self.N_cycle, self.k))

        bias = np.zeros((self.N_cycle, self.k))
        fs_pers = np.zeros((self.N_cycle, self.k))
        x_pred = np.zeros((self.N_cycle, self.k, 256, 256, 1))
        truth =  np.zeros((self.N_cycle, self.k, 256, 256, 1))
        i = 0
        test_dataset = self.get_dataset(TEST_FILENAMES, batch_size=1440)
        x, y = next(iter(test_dataset))
        loaded_s = self.model.load_weights(self.path + "model")
        for i in trange(self.N_cycle):
          
            
            fs_res, fs_pers_res,bias_res, xpred = self.FS(x[i:i+ 2 * self.k])
            bias[i] = bias_res
            fs[i] = fs_res
            fs_pers[i] = fs_pers_res
            if self.save_pred == True :
                tru = x[i:i+2*self.k:2,:,:,0]
                np.save(self.path+'cycle_'+str(self.N_cycle) +'_truth_'+str(i)+'.npy', tru)
            #if self.save_pred == True :
                np.save(self.path +'cycle_'+str(self.N_cycle) +'_pred_'+str(i)+'.npy', xpred)
        return fs, fs_pers, bias



