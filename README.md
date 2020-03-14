# Atlas Autoencoders


ML-compression of ATLAS trigger jet events using autoencoders, with the PyTorch and fastai python libraries.

Given 4-dimensional data our objective is to use autoencoder network to reduce the dataset to 3 dimensions, for easy storage due to less memory consumption.


[Setup](#setup)

[Run](#run)

[Quick guide](#quick-guide)

[Data extraction](#data-extraction)

[Analysis](#analysis)

[Saving back uncompressed data](#saving-back-uncompressed-data)

## Setup:
#### Running the container:
Pull the docker container containing useful libraries:
`docker pull atlasml/ml-base`

Run an interactive bash shell in the container, allowing the hostmachine to open jupyter-notebooks running in the container. The port 8899 can be changed if it is already taken.
`docker run -it -p 8899:8888 atlasml/ml-base`

#### To install the project:

From inside the container, pull the project from the git repository.
```
git init
git pull https://github.com/purugupta99/atlas-task
```

Alternatively, from the hostmachine, use `docker cp` to copy the project files and data into the container. (`docker cp` can be very useful to transfer data to or from the hostmachine.)

Install dependencies (from inside the container):
`pip3 install fastai`
`pip3 install hwcounter`

With jupyter-notebook running, one can access it on the hostmachine from the URL localhost:8899


## Run
- To run the notebook install all the dependencies as given under 'Setup' tab or use Google Colab. Put the train and test pickle file in the same folder as the Notebook and run all cell
- For more information regarding each function definition, refer to in-code comments

## Quick guide
**Pre-processing:** Not required as the processed data is already available in `all_jets_test_4D_100_percent.pkl` and `all_jets_train_4D_100_percent.pkl` files

The data comes in 4 dimensions.

**Training:** Training is done in all the notebook of this project for 1000 epochs, to get the required weights -> `learn.fit(1000, lr=lr, wd=wd)` 

**Analysis and plots:** An example of running a 4-dimensional network is `tanh_normal.ipynb`

**Code structure:** The folder named `Other Variants/`, holds training + analysis scripts for different activation functions and regularisations.

|File|Activation|Regularisation|
|:---:|:---:|:---:|
|tanh_normal.ipynb|tanh|weight decay|
|relu.ipynb|relu|weight decay|
|sigmoid.ipynb|sigmoid|weight decay|
|softmax.ipynb|softmax|weight decay|
|tanh_regularised.ipynb|tanh|L2 Regularisation|

## Data extraction
The data is read from the train and test files, `all_jets_train_4D_100_percent.pkl` and `all_jets_test_4D_100_percent.pkl` respectively, giving out the 4 featues as :-

|Features|
|:---:|
|m|
|pt|
|phi|
|eta|

These values are nice to work with since they are not lists of variable length, which suits our networks with constant input sizes.

We are using every jet as a single event as given in the pickle files.

## Analysis
fastai saves trained models in the folder `models/` relative to the training script, with the .pth file extension. 

In `*.ipynb` there is analysis of a network with a 3D latent space (i.e. a 4/3 compression ratio), with histogram comparisons of the different values and residual plots. Special attention might be given to these residuals as they tell a lot about the performance of the network.

## Saving back uncompressed data
To save a 4-dim multi-dimensional array of decoded data, we use numpy.save on the decoded numpy array to get `reduced_features.npy` file.

