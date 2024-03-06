import numpy as np
import tensorflow_addons as tfa
import numpy.ma as ma
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from functools import partial


def load_dataset(filenames, timestep, labeled=True):
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset.element_spec
    if timestep == 4:
        dataset = dataset.map(partial(read_tfrecord4))
    elif timestep == 3:
        dataset = dataset.map(partial(read_tfrecord3))
    elif timestep == 2:
        dataset = dataset.map(partial(read_tfrecord2))
    elif timestep == 1:
        dataset = dataset.map(partial(read_tfrecord1))
    return dataset


def read_tfrecord1(example):

    tfrecord_format = {
        "inputs": tf.io.FixedLenFeature([4 * 256 * 256 * 4], tf.float32),
        "outputs": tf.io.FixedLenFeature([2 * 256 * 256], tf.float32),
    }

    example = tf.io.parse_single_example(example, tfrecord_format)

    inputs = tf.cast(example["inputs"], tf.float32)

    inputs = tf.reshape(inputs, [*[4, 256, 256, 4]])
    inputs = tf.transpose(inputs, [1, 2, 0, 3])
    inputs_forcings = inputs[:, :, 1:, -2:]
    inputs_forcings = tf.reshape(inputs_forcings, [*[256, 256, 6]])
    inputs_past = inputs[:, :, :, 1]
    inputs_past = tf.reshape(inputs_past, [*[256, 256, 4]])
    inputs = tf.concat([inputs_past, inputs_forcings], axis=2)
    outputs = tf.cast(example["outputs"], tf.float32)
    inputs = replacenan(inputs)
    inputs = inputs + tf.random.normal((256, 256, 10), 0, 0.01)
    outputs = tf.reshape(outputs, [*[256, 256, 2]])
    return inputs, outputs


def read_tfrecord2(example):

    tfrecord_format = {
        "inputs": tf.io.FixedLenFeature([4 * 256 * 256 * 4], tf.float32),
        "outputs": tf.io.FixedLenFeature([2 * 256 * 256], tf.float32),
    }

    example = tf.io.parse_single_example(example, tfrecord_format)

    inputs = tf.cast(example["inputs"], tf.float32)

    inputs = tf.reshape(inputs, [*[4, 256, 256, 4]])
    inputs = tf.transpose(inputs, [1, 2, 0, 3])
    inputs_forcings = inputs[:, :, 1:, -2:]
    inputs_forcings = tf.reshape(inputs_forcings, [*[256, 256, 6]])
    inputs_past = inputs[:, :, :, :2]
    inputs_past = tf.reshape(inputs_past, [*[256, 256, 4 * 2]])
    inputs = tf.concat([inputs_past, inputs_forcings], axis=2)
    outputs = tf.cast(example["outputs"], tf.float32)
    inputs = replacenan(inputs)
    outputs = tf.reshape(outputs, [*[256, 256, 2]])

    return inputs, outputs


def replacenan(t):
    return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)


def get_dataset(timestep, filenames, batch_size):
    print("load dataset")
    dataset = load_dataset(filenames, timestep)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


class Train:
    '''
    Python class to train a surrogate model
    '''
    def __init__(self, model, mask, batch_size, input_shape, lambda_, mu, nu, PATH_TO_SAVE, PATH_TO_DATA, loss, n_filters, 
			timestep, noise = 0, patience = 30, N_output = 1, weight_init = 5e-5, SE_prob = 0, kernel_size = 3, activation = 'relu', dropout_prob = 0.2, epochs = 500, verbose = False, learning_rate = 1e-4 ):
        '''
        Parameters :
        -------------------
        model : Tensorflow model initialized with its subclass
        mask : 2D array with 0 for land pixel and 1 for sea/ice
        
        '''
        self.model = model
        self.mask = mask
        self.batch_size = batch_size
	self.input_shape = input_shape
	self.lambda_ = lambda_
	self.mu_variance = mu_variance
	self.nu_tikhonov = nu_tikhonov
	self.PATH_TO_SAVE = PATH_TO_SAVE
	self.PATH_TO_DATA = PATH_TO_DATA
	self.loss = loss
	self.n_filters = n_filters
	self.timestep = timestep
	self.noise = noise
	self.patience = patience
	self.N_output = N_output
	self.weight_init = weight_init
	self.SE_prob = SE_prob
	self.kernel_size = kernel_size
	self.activation = activation
	self.dropout_prob = dropout_prob
	self.epochs = epochs
	self.verbose = verbose
	self.learning_rate
        



	TRAINING_FILENAMES = tf.io.gfile.glob(self.PATH_TO_DATA + "train_20*")
	VALID_FILENAMES = tf.io.gfile.glob(self.PATH_TO_DATA + "val*")
	TEST_FILENAMES = tf.io.gfile.glob(self.PATH_TO_DATA + "test*")
	print("Train TFRecord Files:", len(TRAINING_FILENAMES))

 	train_dataset = get_dataset(timestep, TRAINING_FILENAMES, batch_size)
	valid_dataset = get_dataset(timestep, VALID_FILENAMES, batch_size)
  	test_dataset = get_dataset(timestep, TEST_FILENAMES, batch_size)
	

	tf.config.run_functions_eagerly(True)
	print("### Define Model ###")
    	model = self.model(
        	mask=self.mask,
        	input_shape=self.input_shape,
        	kernel_size=self.kernel_size,
        	activation=self.activation,
        	depth=3,
        	N_output=self.N_output,
        	SE_prob=self.SE_prob,
        	n_filter=self.n_filters,
        	dropout_prob=self.dropout_prob,
    		)

	# Define optimizer with its learning rate
	step = tf.Variable(0, trainable=False)
    	schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        	[10000, 20000], [1e-0, 1e-1, 1e-2]
   		 )
    	# lr and wd can be a function or a tensor
    	lr = learning_rate * schedule(step)
    	wd = lambda: 1e-6 * schedule(step)
	lambda2 = self.lambda_

    	opt = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    	# Define callbacks
    
	early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience)
    	reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        	monitor="val_loss", factor=0.8, patience=15
    		)
    	modelcheck = tf.keras.callbacks.ModelCheckpoint(
        	path_to_save + "model",
        	monitor="val_loss",
        	verbose=0,
        	save_best_only=True,
        	save_weights_only=True,
    		)

    	def loss_MSE_mask(self, y_true, y_pred):
        	y_true_masked = self.mask * y_true
        	y_pred_masked = self.mask * y_pred
        	loss = 1 / 2 * K.mean(((y_true_masked - y_pred_masked) ** 2))
        	return loss

    	def local_loss(self, y_true, y_pred):
        	y_true_masked = self.mask * y_true
        	y_pred_masked = self.mask * y_pred
        	loss = K.mean(((y_true_masked - y_pred_masked) ** 2))
        	return loss

    	def global_loss(self, y_true, y_pred):
        	y_true_masked = self.mask * y_true
        	y_pred_masked = self.mask * y_pred
        	sum_y_true = K.mean(y_true_masked)
        	sum_y_pred = K.mean(y_pred_masked)
        	loss = K.mean((sum_y_true - sum_y_pred) ** 2)
        	return loss

    	def global_variance(self, y_true, y_pred):
        	y_true_masked = self.mask * y_true
        	y_pred_masked = self.mask * y_pred
        	sum_y_true = K.std(y_true_masked)
        	sum_y_pred = K.std(y_pred_masked)
        	loss = K.mean((sum_y_true - sum_y_pred) ** 2)
        	return loss

	def tikhonov(self, y_true, y_pred):
                y_pred_masked = self.mask * y_pred
                return (y_pred_masked ** 2)

    	def loss_variance(self, y_true, y_pred):

        	y_true_masked = self.mask * y_true
        	y_pred_masked = self.mask * y_pred
        	sum_y_true = K.mean(y_true_masked)
        	sum_y_pred = K.mean(y_pred_masked)

        	std_y_true = K.std(y_true_masked)
        	std_y_pred = K.std(y_pred_masked)
        	loss = (
            		K.mean(((y_true_masked - y_pred_masked) ** 2))
            		+ self.lambda_ * (K.mean((sum_y_true - sum_y_pred) ** 2))
            		+ self.mu_variance * (K.std((y_true_masked - y_pred_masked) ** 2))
			+ self.nu_tikhonov * (y_pred_masked ** 2)
        		)
        	return loss

        loss_mask = loss_variance
    	# Compile the model
    	model.compile(
        	optimizer=opt,
        	loss=loss_mask,
        	metrics=[local_loss, global_loss, global_variance, tikhonov],
    			)

    	history = model.fit(
        	train_dataset,
        	batch_size=self.batch_size,
        	validation_data=valid_dataset,
        	epochs=self.epochs,
        	callbacks=[early_stop, modelcheck],
        	verbose=self.verbose,
	    	)

    	print("### SAVE RESULTS ###")


        np.save(path_to_save + "global_loss.npy", history.history["global_loss"])
        np.save(path_to_save + "local_loss.npy", history.history["local_loss"])
        np.save(path_to_save + "loss.npy", history.history["loss"])
        np.save(
            path_to_save + "global_variance.npy", history.history["global_variance"]
        )
        np.save(
            path_to_save + "val_global_variance.npy",
            history.history["val_global_variance"],
        )
        np.save(path_to_save + "val_loss.npy", history.history["val_loss"])
        np.save(
            path_to_save + "val_global_loss.npy", history.history["val_global_loss"]
        )
        np.save(path_to_save + "val_local_loss.npy", history.history["val_local_loss"])
        )
