# nextsim_surrogate
nextsim fully data-driven surrogate model


### Installation
To install environment: 

```bash
conda env create -f environment.yml
```

To activate the environment:
```bash
conda activate env
```

### Dataset Build

Original Files are download from nextsim NANUK outputs [[1]](#1). Those files are available through the SASIP github. [link to neXtSIM outputs](https://github.com/sasip-climate/catalog-shared-data-SASIP/blob/main/outputs/NANUK025.md). Forcings were dowloaded from ERA5 file [link](https://cds.climate.copernicus.eu/#!/home)

Dataset is build and save under netCDF file with make_dataset script.
```bash
python make_dataset.py
```
To build the dataset with a TFRecord architecture (), use make_tfrecord script
```bash
python make_tfrecord.py
```

Note, the dataset takes a susbstantial amount of place on disk: 360Go for the .nc file, 270Go as .tfrecords

## Run the code

The code is separated on two main section : data and src
- the data directory contains the code to build the dataset, once the original neXtSIM and ERA5 files are downloaded.
- the src file contains all the code to train the neurall network, and compute the test metrics.

To train the neural network:
```bash
python train.py
```
It takes around 18h on 1 GPU A100.
All test metrics are computed with test.py and the Power Spectral Density is compute with 
```bash 
python compute_PSD.py
```

Code to plot the figure in the article is in the plot_article notebook.
