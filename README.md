# GPCNN

Use Gaussian processes (GPs) to add uncertainty to convolutional neural networks (CNNs). GPCNN uses pretrained CNNs as feature extractors to train a GP classifier. The uncertainty in the CNN predictions is determined using the variance associated with the prediction made by the GP classifier, as opposed to use the outputs of the CNN to determine the uncertainty directly.

## Setup

**NOTE** This implementation is not compatible with the latest version of GPyTorch.

Create a virtual environment, for example using Anaconda:

> conda create -n gpcnn python=3.6

Activate the virtual environment:

> conda activate gpcnn

Install the requirements:

> pip install -r requirements.txt

## Experiments

For all experiments, the hyperparameters can be adjusted in the `settings.yaml` file associated with each model in the `models/` directory.

### MNIST

#### Running

First train the CNN directly:

> python main.py -e mnist -m mnist

Thereafter, train the GP classifier using the pretrained MNIST model as a feature extractor:

> python main.py -e mnist_gpcnn -m gpcnn -g mnist

Results can be generated using the following command:

> python main.py -e mnist_gpcnn -m gpcnn -g mnist --evaluate

#### Results

The following are examples of the distributions of the standard deviations associated with each class, when making correct and incorrect predictions:

<img src="https://www.dropbox.com/s/2eom7c6afy5gohp/class_0.png?raw=1" width="300"/> <img src="https://www.dropbox.com/s/mddg2906fdbx91y/class_4.png?raw=1" width="300"/> <img src="https://www.dropbox.com/s/qfmfxg80zrzxo4w/class_5.png?raw=1" width="300"/>

The following are examples of the variance associated with the probability of a prediction being correct for a specific class:

<img src="https://www.dropbox.com/s/yn18rum27kb8h72/class_0.png?raw=1" width="300"/> <img src="https://www.dropbox.com/s/b521rsl59f88xiv/class_4.png?raw=1" width="300"/> <img src="https://www.dropbox.com/s/b56jysopz2n5mou/class_6.png?raw=1" width="300"/>


### noisy-MNIST

TODO

### CIFAR10

TODO
