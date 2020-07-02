# DLWP-CS: Deep Learning Weather Prediction

#### DLWP-CS is a Python project containing data-processing and model-building tools for predicting the gridded atmosphere using deep convolutional neural networks applied to a cubed sphere.

## Reference

If you use this code or find it useful please cite [our publication](https://arxiv.org/abs/2003.11927) (peer-reviewed manuscript coming soon to JAMES).

## Getting started

For now, DLWP is not a package that can be installed using `pip` or a `setup.py` file, so it works like most research code: download (or clone) and run.

#### Required dependencies

It is assumed that the following are installed using Anaconda Python 3 (Python 2.7 should be supported, but not recommended). 
I highly recommend creating a new environment, e.g., `conda create --name dlwp python=3.7 ipython`, and changing to that new environment.

- [TensorFlow](https://www.tensorflow.org) >= 2.0 (GPU capable version highly recommended). 
The `conda` package, while not the recommended installation method, is easy and also installs the required CUDA dependencies. 
If you already have CUDA 10.1 installed, then use `pip install tensorflow`.
For best performance, follow the instructions for installing from source.   
  `conda install tensorflow`
- netCDF4  
  `conda install netCDF4`
- [xarray](http://xarray.pydata.org/en/stable/) with dask; also installs pandas  
  `conda install dask xarray`
- [tempest-remap](https://github.com/ClimateGlobalChange/tempestremap): for cubed-sphere remapping. 
May be installed with conda-forge, but the latest version on GitHub is capable of producing map files from input netCDF files.  
  `conda install -c conda-forge tempest-remap`

#### Optional dependencies

The following are required only for some of the DLWP features:

- [PyTorch](https://pytorch.org): for torch-based deep learning models. 
Again the GPU-ready version is recommended.  
  `pip install torch torchvision`
- [scikit-learn](https://scikit-learn.org/stable/): for machine learning pre-processing tools such as Scalers and Imputers  
  `conda install scikit-learn`
- scipy: for raw data interpolation
- pygrib: for raw CFS data processing  
  `pip install pygrib`
- cdsapi: for retrieval of ERA5 data  
  `pip install cdsapi`
- pyspharm: spherical harmonics transforms for the barotropic model  
  `conda install -c conda-forge pyspharm`

## Quick overview

### General framework

DLWP is built as a weather forecasting model that can, should performance improve greatly, "replace" and existing global weather or climate model. 
Essentially, this means that DLWP uses a deep convolutional neural network to map the state of the atmosphere at one time to the entire state of the atmophere at the next available time. 
A continuous forecast can then be made by feeding the model's predicted state back in as inputs, producing indefinite forecasts.

### Data processing

The classes in `DLWP.data` provide tools for retrieving and processing raw data from the [CFS reanalysis and reforecast](https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/climate-forecast-system-version2-cfsv2) and the [ERA5 reanalysis](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5). 
Meanwhile, the `DLWP.model.preprocessing` module provides tools for formatting the data for ingestion into the deep learning models. 
The first tutorial, "1 - Downloading and processing ERA5", illustrates how to leverage the classes in these modules to prepare data for a DLWP model.
The resulting file of predictor data can be ingested into the data generators for the models.

### Cubed sphere mapping

The `CubeSphereRemap` class in `DLWP.remap` provides functionality for using the `tempest-remap` package for remapping predictor data to a cubed sphere. 
See the tutorial "2 - Remapping to the cubed sphere" for example usage.

### Keras models

The `DLWP.model` module contains classes for building and training TensorFlow/Keras and PyTorch models. 
The `DLWPNeuralNet` class is essentially a wrapper for the simple Keras `Sequential` model, adding optional run-time scaling and imputing of data. 
It implements a few key methods:

- `build_model`: use a custom API to assemble layers in a `Sequential` model. 
Also implements models running on multiple GPUs.  
- `fit`: scale the data and fit the model  
- `fit_generator`: use the Keras `fit_generator` method along with a custom data generator (see section below). 
TensorFlow has officially deprecated the `fit_generator` method so it may be modified in the future.  
- `predict`: predict with the model  
- `predict_timeseries`: predict a continuous time series forecast, where the output of one prediction iteration is used as the input for the next  

DLWP also implements a `DLWPFunctional` class which implements the same methods as the `DLWPNeuralNet` class but takes as input to `build_model` a model assembled using the Keras functional API. 
See the tutorial "3 - Training a DLWP-CS model" for an example of training a model using the `DLWPFunctional` class.

### PyTorch models

Currently, due to a focus on TensorFlow/Keras models, the PyTorch implementation in DLWP is more limited, although still robust. 
Like the Keras models, it implements a convenient `build_model` method to assemble a sequential-like model using the same API parameters as those for `DLWPNeuralNet`. 
Additionally, it also implements a `fit` method to automatically iterate through the data and optimizer, again, just like the Keras API. 
However, a separate class for functional-type general models has not yet been developed.

### Custom layers and functions

The `DLWP.custom` module contains many custom layers specifically for applying convolutional neural networks to global weather on the cubed sphere.  
`CubeSpherePadding2D` implements padding across cube faces prior to applying convolutions. 
The convolutional layer for data on a cubed sphere is `CubeSphereConv2D`. 
It works just like standard Keras API layers and has options for specifying unique weights for the polar faces
These custom layers are worth a look.

### Data generators

`DLWP.model.generators` contains several classes for generating data on-the-fly from a netCDF file produced by the DLWP preprocessing methods. 
These data generators can then be used in conjunction with a DWLP model instance's `fit_generator` method.
- The `DataGenerator` class is the simplest generator class. 
It merely returns batches of data from a file containing "predictors" and "targets" variables already formatted for use in the DLWP model. 
Due to this simplicity, this is the optimal way to generate data directly from the disk when system memory is not sufficient to load the entire dataset. 
However, this comes at the cost of generating very large files on disk with redundant data (since the targets are merely a different time shift of the predictors).
- The `SeriesDataGenerator` class is much more robust and memory efficient. 
It expects only a single "predictors" variable in the input file and generates predictor-target pairs on the fly for each batch of data. 
It also has the ability to prescribe external fields such as incoming solar radiation and constants read from files. 
- The `ArrayDataGenerator` is a slightly different version of `SeriesDataGenerator` which uses a single `numpy` array of data instead of an `xarray` `Dataset`. 
While this was designed to be marginally faster, there is little practical benefit. 
The array and auxiliary parameters for this generator can be produced with the `DLWP.model.preprocessing.prepare_data_array` method.

Since TensorFlow 2.0 doesn't play nicely with `multiprocessing` (memory leaks), it is not recommended to use the Keras API multiprocessing feature. 
Instead, I recommend creating a `tensorflow.data.Dataset` to feed into the model `fit` method. 
This can be done with the `tf_data_generator` wrapper. 
Pass one of the above data generators to this method.


### Advanced forecast tools

The `DLWP.model` module also contains a `TimeSeriesEstimator` class. 
This class can be used to make robust forward forecasts where the data input does not necessarily match the data output of a model.
This is the recommended way of making iterative predictions; see the tutorial "4 - Predicting with a DLWP-CS model".

### Other

The `DLWP.util` module contains useful utilities, including `save_model` and `load_model` for saving and loading DLWP models (and correctly dealing with multi-GPU models).
