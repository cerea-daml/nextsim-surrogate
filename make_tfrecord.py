import numpy as np
from tqdm import trange
import numpy.ma as ma
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import xarray as xr


def np_to_tfrecords(X, Y, file_path_prefix, verbose=True):
    """
    Converts a Numpy array (or two Numpy arrays) into a tfrecord file.
    For supervised learning, feed training inputs to X and training labels to Y.
    For unsupervised learning, only feed training inputs to X, and feed None to Y.
    The length of the first dimensions of X and Y should be the number of samples.
    
    Parameters
    ----------
    X : numpy.ndarray of rank 2
        Numpy array for training inputs. Its dtype should be float32, float64, or int64.
        If X has a higher rank, it should be rshape before fed to this function.
    Y : numpy.ndarray of rank 2 or None
        Numpy array for training labels. Its dtype should be float32, float64, or int64.
        None if there is no label array.
    file_path_prefix : str
        The path and name of the resulting tfrecord file to be generated, without '.tfrecords'
    verbose : bool
        If true, progress is reported.
    
    Raises
    ------
    ValueError
        If input type is not float (64 or 32) or int.
    
    """

    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(
                float_list=tf.train.FloatList(value=array)
            )
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(
                int64_list=tf.train.Int64List(value=array)
            )
        else:
            raise ValueError(
                "The input should be numpy ndarray. \
                               Instaed got {}".format(
                    ndarray.dtype
                )
            )

    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 2  # If X has a higher rank,
    # it should be rshape before fed to this function.
    assert isinstance(Y, np.ndarray) or Y is None

    # load appropriate tf.train.Feature class depending on dtype
    dtype_feature_x = _dtype_feature(X)

    assert X.shape[0] == Y.shape[0]
    assert len(Y.shape) == 2
    dtype_feature_y = _dtype_feature(Y)

    # Generate tfrecord writer
    result_tf_file = file_path_prefix + ".tfrecords"
    writer = tf.io.TFRecordWriter(result_tf_file)
    if verbose:
        print("Serializing {:d} examples into {}".format(X.shape[0], result_tf_file))

    # iterate over each sample,
    # and serialize it as ProtoBuf.
    for idx in trange(X.shape[0]):
        x = X[idx]

        y = Y[idx]

        d_feature = {}
        d_feature["inputs"] = dtype_feature_x(x)

        d_feature["outputs"] = dtype_feature_y(y)

        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)

    if verbose:
        print("Writing {} done!".format(result_tf_file))


def split_tfrecord(tfrecord_path, split_size):
    '''
    divide a tfrecords file into chunks 
    
    Params
    -----------------
    tfrecord_path : str, path to the tfrecord file
    split_size : int, number of example in each file
    '''
    with tf.Graph().as_default(), tf.Session() as sess:
        ds = tf.data.TFRecordDataset(tfrecord_path).batch(split_size)
        batch = ds.make_one_shot_iterator().get_next()
        part_num = 0
        while True:
            try:
                records = sess.run(batch)
                part_path = tfrecord_path + ".{:03d}".format(part_num)
                with tf.python_io.TFRecordWriter(part_path) as writer:
                    for record in records:
                        writer.write(record)
                part_num += 1
            except tf.errors.OutOfRangeError:
                break

def split_tf_record()
    data = xr.open_dataset("val_input.nc")

    data = data.drop_dims(["x", "y"])
    data = data.to_array().to_numpy()
    X = data
    X = np.swapaxes(X, 0, 1)
    X = X.reshape((np.shape(X)[0], -1))

    Y = xr.open_dataset("yval_norm.nc")
    Y = Y.drop_dims(["x", "y"])
    Y = Y.to_array().to_numpy()
    Y = np.swapaxes(Y, 0, 1)
    Y = Y.reshape((np.shape(Y)[0], -1))

    np_to_tfrecords(X, Y, "val", verbose=True)
    split_tfrecord("val.tfrecords", 60)

    data = xr.open_dataset("test_input.nc")

    data = data.drop_dims(["x", "y"])
    data = data.to_array().to_numpy()

    X = data
    X = np.swapaxes(X, 0, 1)
    X = X.reshape((np.shape(X)[0], -1))
    X = X[:-1]
    Y = xr.open_dataset("ytest_norm.nc")
    Y = Y.drop_dims(["x", "y"])
    Y = Y.to_array().to_numpy()
    Y = np.swapaxes(Y, 0, 1)
    Y = Y.reshape((np.shape(Y)[0], -1))

    np_to_tfrecords(X, Y, "test", verbose=True)
    split_tfrecord("test.tfrecords", 60)


    data = xr.open_dataset("2009_input.nc")
    data = data.drop_dims(["x", "y"])
    data = data.to_array().to_numpy()
    X = data
    X = np.swapaxes(X, 0, 1)
    X = X.reshape((np.shape(X)[0], -1))

    Y = xr.open_dataset("2009_output.nc")
    Y = Y.drop_dims(["x", "y"])
    Y = Y.to_array().to_numpy()
    Y = np.swapaxes(Y, 0, 1)
    Y = Y.reshape((np.shape(Y)[0], -1))

    np_to_tfrecords(X, Y, "train_2009", verbose=True)
    split_tfrecord("train_2009.tfrecords", 60)


    data = xr.open_dataset("2010_input.nc")
    data = data.drop_dims(["x", "y"])
    data = data.to_array().to_numpy()
    X = data
    X = np.swapaxes(X, 0, 1)
    X = X.reshape((np.shape(X)[0], -1))

    Y = xr.open_dataset("2010_output.nc")
    Y = Y.drop_dims(["x", "y"])
    Y = Y.to_array().to_numpy()
    Y = np.swapaxes(Y, 0, 1)
    Y = Y.reshape((np.shape(Y)[0], -1))

    np_to_tfrecords(X, Y, "train_2010", verbose=True)
    split_tfrecord("train_2010.tfrecords", 60)


    data = xr.open_dataset("2011_input.nc")
    data = data.drop_dims(["x", "y"])
    data = data.to_array().to_numpy()
    X = data
    X = np.swapaxes(X, 0, 1)
    X = X.reshape((np.shape(X)[0], -1))

    Y = xr.open_dataset("2011_output.nc")
    Y = Y.drop_dims(["x", "y"])
    Y = Y.to_array().to_numpy()
    Y = np.swapaxes(Y, 0, 1)
    Y = Y.reshape((np.shape(Y)[0], -1))

    np_to_tfrecords(X, Y, "train_2011", verbose=True)
    split_tfrecord("train_2011.tfrecords", 60)


    data = xr.open_dataset("2012_input.nc")
    data = data.drop_dims(["x", "y"])
    data = data.to_array().to_numpy()
    X = data
    X = np.swapaxes(X, 0, 1)
    X = X.reshape((np.shape(X)[0], -1))

    Y = xr.open_dataset("2012_output.nc")
    Y = Y.drop_dims(["x", "y"])
    Y = Y.to_array().to_numpy()
    Y = np.swapaxes(Y, 0, 1)
    Y = Y.reshape((np.shape(Y)[0], -1))

    np_to_tfrecords(X, Y, "train_2012", verbose=True)
    split_tfrecord("train_2012.tfrecords", 60)


    data = xr.open_dataset("2013_input.nc")
    data = data.drop_dims(["x", "y"])
    data = data.to_array().to_numpy()
    X = data
    X = np.swapaxes(X, 0, 1)
    X = X.reshape((np.shape(X)[0], -1))

    Y = xr.open_dataset("2013_output.nc")
    Y = Y.drop_dims(["x", "y"])
    Y = Y.to_array().to_numpy()
    Y = np.swapaxes(Y, 0, 1)
    Y = Y.reshape((np.shape(Y)[0], -1))

    np_to_tfrecords(X, Y, "train_2013", verbose=True)
    split_tfrecord("train_2013.tfrecords", 60)


    data = xr.open_dataset("2014_input.nc")
    data = data.drop_dims(["x", "y"])
    data = data.to_array().to_numpy()
    X = data
    X = np.swapaxes(X, 0, 1)
    X = X.reshape((np.shape(X)[0], -1))

    Y = xr.open_dataset("2014_output.nc")
    Y = Y.drop_dims(["x", "y"])
    Y = Y.to_array().to_numpy()
    Y = np.swapaxes(Y, 0, 1)
    Y = Y.reshape((np.shape(Y)[0], -1))

    np_to_tfrecords(X, Y, "train_2014", verbose=True)
    split_tfrecord("train_2014.tfrecords", 60)


    data = xr.open_dataset("2015_input.nc")
    data = data.drop_dims(["x", "y"])
    data = data.to_array().to_numpy()
    X = data
    X = np.swapaxes(X, 0, 1)
    X = X.reshape((np.shape(X)[0], -1))

    Y = xr.open_dataset("2015_output.nc")
    Y = Y.drop_dims(["x", "y"])
    Y = Y.to_array().to_numpy()
    Y = np.swapaxes(Y, 0, 1)
    Y = Y.reshape((np.shape(Y)[0], -1))

    np_to_tfrecords(X, Y, "train_2015", verbose=True)
    split_tfrecord("train_2015.tfrecords", 60)


    data = xr.open_dataset("2016_input.nc")
    data = data.drop_dims(["x", "y"])
    data = data.to_array().to_numpy()
    X = data
    X = np.swapaxes(X, 0, 1)
    X = X.reshape((np.shape(X)[0], -1))

    Y = xr.open_dataset("2016_output.nc")
    Y = Y.drop_dims(["x", "y"])
    Y = Y.to_array().to_numpy()
    Y = np.swapaxes(Y, 0, 1)
    Y = Y.reshape((np.shape(Y)[0], -1))

    np_to_tfrecords(X, Y, "train_2016", verbose=True)
    split_tfrecord("train_2016.tfrecords", 60)


split_tf_record()
