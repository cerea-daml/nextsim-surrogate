#!/usr/bin/env python3


import xarray as xr
import numpy as np
import numpy.ma as ma
from tqdm import trange
from skimage.measure import block_reduce
from scipy.spatial import cKDTree

###################################
# Python script for the generation of data over NextSIM Simulation results for 2018 #
###################################


class create_dataset_from_nextsim_outputs:
    """
    Build a dataset store into netcdf4 files from Nextsim outputs and ERA5 forcings
    """

    def __init__(self, N_res, PATH_NEXTSIM, PATH_FORCINGS, reduce_func=np.mean):
        """
        Build a class to create the sea ice thickess and forcings dataset
        
        --------------------------
        PARAMETERS :
        N_res : int, coefficient ration for the resolution, if N_res = 1, images are 512x512 pixels
	PATH_NEXTSIM : str, directory where nextsim outputs are located
	PATH_FORCINGS : str, directory where nextsim outputs are located
        reduce_func : numpyh function, function to apply to use in the skimage.measure.block_reduce function
       
        """
        self.N_res = N_res
        self.reduce_func = reduce_func
	self.PATH_NEXTSIM = PATH_NEXTSIM
	self.PATH_FORCINGS = PATH_FORCINGS

    def prepare_data(self, x):
	"""return a np array with the variable wanted at the correct shape and the correct resolution
	-------------------------
	PARAMETERS : 
	x : xarray DataArray of the physical variable of interest
	"""

	#Get data from masked array
        N = np.shape(x)[0]
        x_t = []
        for i in trange(N):
            x_t.append(ma.getdata(x[i]).reshape((603, 528, 1)))

        N_shape = int(512 / self.N_res)
        x_t = np.array(x_t)

        #Select the 512x512 pixels
        x_t = x_t[:, 91:, 8:-8]


	#Get mask, reshape and rescale
        mask = ma.getmaskarray(x[0])
        mask = mask.reshape((603, 528, 1))
        mask = mask[91:, 8:-8]
        mask2 = block_reduce(mask, block_size=(self.N_res, self.N_res, 1), func=np.min)

	#Replace masked values with 0
        x_t = np.where(mask == True, 0, np.array(x_t))

	#If we coarsen the images : image rescaling with averaging over pixels
        if self.N_res > 1:
            x_coarse = np.zeros((np.shape(x_t)[0], N_shape, N_shape, 1))

            for i in trange(np.shape(x_t)[0]):
                x_coarse[i] = block_reduce(
                    x_t[i],
                    block_size=(self.N_res, self.N_res, 1),
                    func=self.reduce_func,
                )

            x_t = x_coarse
        return x_t, mask2

    def build_variable(self):
	
        N_shape = int(512 / self.N_res)

        liste = []
        years_train = ["2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016"]
        months = [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ]
        for y in years_train:
            for m in months:
                liste.append("Data/Moorings_" + y + "m" + m + ".nc")
        print("Build Train")
        ds_train = xr.open_mfdataset(liste)
        sit_train = ds_train.sit
        time_train = ds_train.time
        sit_train = sit_train.to_masked_array()
        print("Build Val")
        ds_val = xr.open_mfdataset("Data/Moorings_2017m*.nc")
        sit_val = ds_val.sit
        time_val = ds_val.time
        sit_val = sit_val.to_masked_array()
        print("Build Test")
        ds_test = xr.open_mfdataset("Data/Moorings_2018m*.nc")
        sit_test = ds_test.sit
        time_test = ds_test.time
        sit_test = sit_test.to_masked_array()

        print("prepare test")
        x_test, mask = self.prepare_data(sit_test)
        print(np.shape(mask))
        print("prepare val")
        x_val, _ = self.prepare_data(sit_val)
        print("prepare train")
        x_train, _ = self.prepare_data(sit_train)

        np.save("mask.npy", mask)
        x_tr = x_train

        y_tr = np.concatenate(
            [x_train[8:-2] - x_train[6:-4], x_train[10:] - x_train[6:-4]], axis=3
        )
        N_tr = np.shape(x_tr)[0]
        x_train = np.zeros((N_tr - 10, N_shape, N_shape, 4))
        y_train = y_tr
        for i in range(N_tr - 10):
            x_train[i, :, :, 0] = x_tr[i].squeeze()
            x_train[i, :, :, 1] = x_tr[i + 2].squeeze()
            x_train[i, :, :, 2] = x_tr[i + 4].squeeze()
            x_train[i, :, :, 3] = x_tr[i + 6].squeeze()
        y_v = np.concatenate(
            [x_val[8:-2] - x_val[6:-4], x_val[10:] - x_val[6:-4]], axis=3
        )
        x_v = x_val

        N_val = np.shape(x_v)[0]
        x_val = np.zeros((N_val - 10, N_shape, N_shape, 4))
        y_val = y_v
        for i in range(N_val - 10):
            x_val[i, :, :, 0] = x_v[i].squeeze()
            x_val[i, :, :, 1] = x_v[i + 2].squeeze()
            x_val[i, :, :, 2] = x_v[i + 4].squeeze()
            x_val[i, :, :, 3] = x_v[i + 6].squeeze()

        y_te = np.concatenate(
            [x_test[8:-2] - x_test[6:-4], x_test[10:] - x_test[6:-4]], axis=3
        )
        x_te = x_test

        N_test = np.shape(x_te)[0]
        x_test = np.zeros((N_test - 10, N_shape, N_shape, 4))
        y_test = y_te
        for i in range(N_test - 10):
            x_test[i, :, :, 0] = x_te[i].squeeze()
            x_test[i, :, :, 1] = x_te[i + 2].squeeze()
            x_test[i, :, :, 2] = x_te[i + 4].squeeze()
            x_test[i, :, :, 3] = x_te[i + 6].squeeze()

        source = xr.open_dataset("Data/Moorings_2009m01.nc")
        lat_source = source.variables["latitude"][91:, 8:-8]
        lon_source = source.variables["longitude"][91:, 8:-8]
        lat_source = lat_source[:: self.N_res, :: self.N_res]
        lon_source = lon_source[:: self.N_res, :: self.N_res]

        mask = np.multiply(mask, 1.0)
        self.mask = mask
        print("VAL INPUTS")
        val1 = xr.Dataset(
            coords={
                "time": time_val[10:],
                "prec": [1, 2, 3, 4],
                "lat": (["x", "y"], lat_source),
                "lon": (["x", "y"], lon_source),
            }
        )
        val1["inputs_sit"] = (["time", "lat", "lon", "prec"], x_val.squeeze())
        val1.to_netcdf("CoarseResolution/val_inputs.nc", mode="w")
        print("VAL OUTPUTS")
        val2 = xr.Dataset(
            coords={
                "time": time_val[10:],
                "prec": [1, 2],
                "lat": (["x", "y"], lat_source),
                "lon": (["x", "y"], lon_source),
            }
        )
        val2["outputs_sit"] = (["time", "lat", "lon", "prec"], y_val.squeeze())
        val2.to_netcdf("CoarseResolution/val_outputs.nc", mode="w")
        print("TEST_INPUTS")
        test1 = xr.Dataset(
            coords={
                "time": time_test[10:],
                "prec": [1, 2, 3, 4],
                "lat": (["x", "y"], lat_source),
                "lon": (["x", "y"], lon_source),
            }
        )
        test1["inputs_sit"] = (["time", "lat", "lon", "prec"], x_test.squeeze())
        test1.to_netcdf("CoarseResolution/test_inputs.nc", mode="w")
        print("TEST OUTPUTS")
        test2 = xr.Dataset(
            coords={
                "time": time_test[10:],
                "prec": [1, 2],
                "lat": (["x", "y"], lat_source),
                "lon": (["x", "y"], lon_source),
            }
        )
        test2["outputs_sit"] = (["time", "lat", "lon", "prec"], y_test.squeeze())
        test2.to_netcdf("CoarseResolution/test_outputs.nc", mode="w")
        print(test2)

        print("Train")

        train1 = xr.Dataset(
            coords={
                "time": time_train[10:],
                "prec": [1, 2, 3, 4],
                "lat": (["x", "y"], lat_source),
                "lon": (["x", "y"], lon_source),
            }
        )
        train1["inputs_sit"] = (["time", "lat", "lon", "prec"], x_train.squeeze())
        train1.to_netcdf("CoarseResolution/train_inputs.nc", mode="w")
        print("VAL OUTPUTS")
        print(train1)
        train2 = xr.Dataset(
            coords={
                "time": time_train[10:],
                "prec": [1, 2],
                "lat": (["x", "y"], lat_source),
                "lon": (["x", "y"], lon_source),
            }
        )
        train2["outputs_sit"] = (["time", "lat", "lon", "prec"], y_train.squeeze())
        train2.to_netcdf("CoarseResolution/train_outputs.nc", mode="w")
        return mask

    def interpolate(self, time, inputs, dataset, data_name, target_shape, d, inds):
        nt = time.shape[0]
        tmp = {}

        tmp[data_name] = []

        for t in trange(0, nt):

            var = dataset[data_name][t].values.flatten()[inds]

            var.shape = target_shape.shape
            var = block_reduce(
                var[91:, 8:-8], block_size=(self.N_res, self.N_res), func=np.mean
            )
            tmp[data_name].append(var)

        return tmp

    def lon_lat_to_cartesian(self, lon, lat):
        # WGS 84 reference coordinate system parameters
        A = 6378.137  # major axis [km]
        E2 = 6.69437999014e-3  # eccentricity squared

        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)
        # convert to cartesian coordinates
        r_n = A / (np.sqrt(1 - E2 * (np.sin(lat_rad) ** 2)))
        x = r_n * np.cos(lat_rad) * np.cos(lon_rad)
        y = r_n * np.cos(lat_rad) * np.sin(lon_rad)
        z = r_n * (1 - E2) * np.sin(lat_rad)
        return x, y, z

    def create_forcings_dataset(self, nextsim_data, variable, data_name):
        source = xr.open_dataset(nextsim_data)
        lat_source = source.variables["latitude"]
        lon_source = source.variables["longitude"]
        print("OPEN FORCINGS DATASET")

        liste = []
        years_train = ["2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016"]
        for y in years_train:
            liste.append("Forcings/ERA5_" + variable + "_y" + y + ".nc")
        dataset_train = xr.open_mfdataset(liste)
        dataset_train.fillna(0)

        time_train = dataset_train.time.data

        dataset_val = xr.open_dataset("Forcings/ERA5_" + variable + "_y2017.nc")
        dataset_val.fillna(0)

        dataset_test = xr.open_dataset("Forcings/ERA5_" + variable + "_y2018.nc")
        dataset_test.fillna(0)

        time_test = dataset_test.time.data
        time_val = dataset_val.time.data

        lat_target = dataset_val.latitude
        lon_target = dataset_val.longitude

        lon_target2d, lat_target2d = np.meshgrid(lon_target, lat_target)

        print("CONVERT LAT, LON TO CARTESIAN GRID")
        xt, yt, zt = self.lon_lat_to_cartesian(
            lon_source.values.flatten(), lat_source.values.flatten()
        )
        xs, ys, zs = self.lon_lat_to_cartesian(
            lon_target2d.flatten(), lat_target2d.flatten()
        )

        print("INTERPOLATE")
        tree = cKDTree(np.column_stack((xs, ys, zs)))
        d, inds = tree.query(np.column_stack((xt, yt, zt)), k=1)

        forcings_train = self.interpolate(
            time_train, source, dataset_train, data_name, lat_source, d, inds
        )
        forcings_val = self.interpolate(
            time_val, source, dataset_val, data_name, lat_source, d, inds
        )
        forcings_test = self.interpolate(
            time_test, source, dataset_test, data_name, lat_source, d, inds
        )

        print("CREATE DATASETS")
        lat_source = lat_source[91:, 8:-8]
        lon_source = lon_source[91:, 8:-8]

        lat_source = lat_source[:: self.N_res, :: self.N_res]
        lon_source = lon_source[:: self.N_res, :: self.N_res]

        print(np.shape(lat_source))
        print(np.shape(lon_source))
        dataset_train = xr.Dataset(
            coords={
                "time": time_train[10:],
                "prec": [1, 2, 3, 4],
                "lat": (["x", "y"], lat_source),
                "lon": (["x", "y"], lon_source),
            }
        )
        dataset_val = xr.Dataset(
            coords={
                "time": time_val[10:],
                "prec": [1, 2, 3, 4],
                "lat": (["x", "y"], lat_source),
                "lon": (["x", "y"], lon_source),
            }
        )

        dataset_test = xr.Dataset(
            coords={
                "time": time_test[10:],
                "prec": [1, 2, 3, 4],
                "lat": (["x", "y"], lat_source),
                "lon": (["x", "y"], lon_source),
            }
        )
        print("Add variables")

        dataset_train[data_name] = (
            ["time", "prec", "lat", "lon"],
            np.array(
                [
                    np.array(forcings_train[data_name])[0:-10],
                    np.array(forcings_train[data_name])[2:-8],
                    np.array(forcings_train[data_name])[4:-6],
                    np.array(forcings_train[data_name])[6:-4],
                ]
            ).transpose((1, 0, 2, 3)),
        )

        dataset_val[data_name] = (
            ["time", "prec", "lat", "lon"],
            np.array(
                [
                    np.array(forcings_val[data_name])[0:-10],
                    np.array(forcings_val[data_name])[2:-8],
                    np.array(forcings_val[data_name])[4:-6],
                    np.array(forcings_val[data_name])[6:-4],
                ]
            ).transpose((1, 0, 2, 3)),
        )
        dataset_test[data_name] = (
            ["time", "prec", "lat", "lon"],
            np.array(
                [
                    np.array(forcings_test[data_name])[0:-10],
                    np.array(forcings_test[data_name])[2:-8],
                    np.array(forcings_test[data_name])[4:-6],
                    np.array(forcings_test[data_name])[6:-4],
                ]
            ).transpose((1, 0, 2, 3)),
        )

        mean = dataset_train[data_name].mean(skipna=True)

        std = dataset_train[data_name].std(skipna=True)

        dataset_train[data_name] = (dataset_train[data_name] - mean) / std

        dataset_val[data_name] = (dataset_val[data_name] - mean) / std
        dataset_test[data_name] = (dataset_test[data_name] - mean) / std

        print(dataset_val[data_name].mean(skipna=True))
        print(dataset_test[data_name].std(skipna=True))
        print("WRITE")
        print(dataset_train)
        dataset_train.to_netcdf(path="./" + data_name + "_train_forcings.nc", mode="w")
        dataset_val.to_netcdf(path="./" + data_name + "_val_forcings.nc", mode="w")
        dataset_test.to_netcdf(path="./" + data_name + "_test_forcings.nc", mode="w")

    def create_forcings(self, path_to_data):
        self.create_forcings_dataset(path_to_data, "t2m", "t2m")
        self.create_forcings_dataset(path_to_data, "sst", "sst")
        self.create_forcings_dataset(path_to_data, "u10", "u10")
        self.create_forcings_dataset(path_to_data, "v10", "v10")

    def normalisation(self, path_to_file):

        print("open train")
        xtrain = xr.open_dataset(path_to_file + "train_inputs.nc")
        ytrain = xr.open_dataset(path_to_file + "train_outputs.nc")
        print("open val")
        xval = xr.open_dataset(path_to_file + "val_inputs.nc")
        yval = xr.open_dataset(path_to_file + "val_outputs.nc")
        print("open test")
        xtest = xr.open_dataset(path_to_file + "test_inputs.nc")
        ytest = xr.open_dataset(path_to_file + "test_outputs.nc")

        mean_input = xtrain.mean(dim=["time", "lat", "lon", "x", "y"])
        std_input = xtrain.std(dim=["time", "lat", "lon", "x", "y"])
        mean_output = ytrain.mean(dim=["time", "lat", "lon", "x", "y"])
        std_output = ytrain.std(dim=["time", "lat", "lon", "x", "y"])

        np.save("mean_input.npy", mean_input["inputs_sit"][0])
        np.save("mean_output.npy", mean_output["outputs_sit"][0])
        np.save("std_input.npy", std_input["inputs_sit"][0])
        np.save("std_output.npy", std_output["outputs_sit"][0])

        mask = self.mask
        print("Normalize input")
        xtrain = (
            (xtrain - mean_input)
            / std_input
            * ((mask == 1).sum() + (mask == 0).sum())
            / (mask == 1).sum()
            * 2
        )
        xval = (
            (xval - mean_input)
            / std_input
            * ((mask == 1).sum() + (mask == 0).sum())
            / (mask == 1).sum()
            * 2
        )
        xtest = (
            (xtest - mean_input)
            / std_input
            * ((mask == 1).sum() + (mask == 0).sum())
            / (mask == 1).sum()
            * 2
        )
        print("Normalize output")
        ytrain = (
            (ytrain - mean_output)
            / std_output
            * ((mask == 1).sum() + (mask == 0).sum())
            / (mask == 1).sum()
            * 2
        )
        yval = (
            (yval - mean_output)
            / std_output
            * ((mask == 1).sum() + (mask == 0).sum())
            / (mask == 1).sum()
            * 2
        )
        ytest = (
            (ytest - mean_output)
            / std_output
            * ((mask == 1).sum() + (mask == 0).sum())
            / (mask == 1).sum()
            * 2
        )
        print("Write train")
        xtrain.to_netcdf("xtrain_norm.nc", mode="w")
        ytrain.to_netcdf("ytrain_norm.nc", mode="w")
        print("Write val")
        xval.to_netcdf("xval_norm.nc", mode="w")
        yval.to_netcdf("yval_norm.nc", mode="w")
        print("write test")
        xtest.to_netcdf("xtest_norm.nc", mode="w")
        ytest.to_netcdf("ytest_norm.nc", mode="w")

    def merge_sit_forcings(self):
        """
        Merge Sea ice thickness variable with ERA5 forcings for training, validation and test dataset
        """
        x1 = xr.open_dataset("xtrain_norm.nc")
        x2 = xr.open_dataset("u10_train_forcings.nc")
        x3 = xr.open_dataset("v10_train_forcings.nc")
        x4 = xr.open_dataset("sst_train_forcings.nc")
        x5 = xr.open_dataset("t2m_train_forcings.nc")

        x = xr.merge([x1, x2, x3, x4, x5])
        x.to_netcdf("train_input.nc", mode="w")

        x1 = xr.open_dataset("xval_norm.nc")
        x2 = xr.open_dataset("u10_val_forcings.nc")
        x3 = xr.open_dataset("v10_val_forcings.nc")
        x4 = xr.open_dataset("sst_val_forcings.nc")
        x5 = xr.open_dataset("t2m_val_forcings.nc")

        x = xr.merge([x1, x2, x3, x4, x5])
        x.to_netcdf("val_input.nc", mode="w")

        x1 = xr.open_dataset("xtest_norm.nc")
        x2 = xr.open_dataset("u10_test_forcings.nc")
        x3 = xr.open_dataset("v10_test_forcings.nc")
        x4 = xr.open_dataset("sst_test_forcings.nc")
        x5 = xr.open_dataset("t2m_test_forcings.nc")

        x = xr.merge([x1, x2, x3, x4, x5])
        x.to_netcdf("test_input.nc", mode="w")

    def split_train_in_years(self):
	"""
	Split the train.nc files by year to ease the writing into tfrecords 
	"""
        x = xr.load_dataset("train_input.nc")

        years, datasets = zip(*x.groupby("time.year"))

        paths = [f"{y}_input.nc" for y in years]

        xr.save_mfdataset(datasets, paths)
        y = xr.load_dataset("ytrain_norm.nc")

        years, datasets = zip(*y.groupby("time.year"))

        paths = [f"{ye}_output.nc" for ye in years]

        xr.save_mfdataset(datasets, paths)


data = create_dataset_from_nextsim_outputs(N_res=8)
data.build_variable()
data.create_forcings("Data/Moorings_2018m08.nc")
data.normalisation("CoarseResolution/")
data.merge_sit_forcings()
data.split_train_in_years()
