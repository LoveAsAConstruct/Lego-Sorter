# LEGO SORTER

## Introduction

This project demonstrates the use of neural networks for image classification and is created for CSCI-e80's week 5 exploratory project. This repository documents my experience with setting up and experimenting with convolutional neural networks.

More information on the assignment can be found [here](https://cs50.harvard.edu/extension/ai/2024/spring/projects/5/exploratory/).
This project's Github repository can be found [here](https://github.com/LoveAsAConstruct/Lego-Sorter)

Before beginning the setup, here are a few important notes:
- You will interact mainly with the `train.py` script. Usage is as follows:
~~~
train.py [-h] [--random] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--data_dir DATA_DIR] [--verbose] [--name NAME] [--save_dir SAVE_DIR] [--runs RUNS]

Train and save lego piece detection models
optional arguments:
-h, --help show this help message and exit
--random Will generate a random model to train, otherwise uses 'ol' reliable'
--epochs EPOCHS Number of training epochs
--batch_size BATCH_SIZE
Batch size
--data_dir DATA_DIR Directory containing lego images
--verbose Enable verbose mode
--name NAME Run name
--save_dir SAVE_DIR Directory to save runtime folder
--runs RUNS Number of runs to test (recommended with random)
~~~
The rest of the project consists of helper Python files containing a model randomization function and constant variables, as well as several folders containing project data. These folders—`archive`, `deprecated`, and `train_results`—store the data used to train the model, deprecated code I implemented in the past, and results from my prior experiences training the model, respectively.

## Setup

### Prerequisites

You will need Conda 24.3.0, which can be downloaded through Miniconda [here](https://docs.anaconda.com/free/miniconda/index.html).
The file downloaded will depend on your system. Once installed, proceed to Environment Setup.

### Environment Setup

Start by activating Conda with the command:
~~~
conda activate
~~~
Then, create the Conda environment with the provided `environment.yml` file:
~~~
conda env create -f environment.yml
~~~
Activate your Conda environment:
~~~
conda activate tf
~~~
This environment should be set up with all the packages needed to run `train.py`. However, if you need to reinstall, `requirements.txt` contains all the packages used, which can be installed through pip:
~~~
pip install -r requirements.txt
~~~

## Usage

The main file of interest is `train.py`, which can be run with or without command-line arguments. Usage is as mentioned in the Introduction section. When recording training sessions, data such as the model's final weights, summary, training parameters, training progress, and overall performance will be saved under the model's name within the `save_dir` storage directory, defaulting to `train_results`. Note that due to the size of the model names, there is a possibility of overflowing the file path character limit, depending on the hierarchy of your computer. If encountering path overflow, I recommend moving the project directory closer to your root.

## Additional Notes

I had a great time experimenting with the classification network! Initially, I had a relatively botched setup, which can be seen in the deprecated folder, with a mishmash of data containers, image preprocessing, and network management. A notable feature that I didn't include in this legacy version is the addition of noise and the variance of brightness and backgrounds. Although it would be interesting to implement within the legacy branch, I felt it would violate the guidelines of prioritizing experimentation with the neural network. So I left that system with the rest of the deprecated files. If implemented, I was hoping that the noise and brightness variation would enhance the detection of real-life Lego pieces, as the current dataset consists of uniform automated renders from Autodesk's Maya 3D modeling program. 

When it comes to the model experimentation on my current dataset, I had some fun playing with parameters! Trained on a dual RTX 3090 setup, I was able to train models in usually less than a minute (depending on the training parameters of course). This enabled me to try many different configurations, ranging from high-density convolutional networks to single-layer attempts. Interestingly, I found that the gain from extended training periods and hidden layers seemed logarithmic, as anything beyond a simple setup I dubbed "ol' faithful" (which is the default model train.py uses) seems to provide relatively similar results. In fact, I was able to train models with 83% accuracy after two epochs with batch sizes of 32, which seemed relatively surprising to me based on the brief training they received. I do suspect that this is due to the uniformity and cleanliness of the dataset, as the 3d renders carry little background noise or difference, making the categorization task easy for the model. 

Despite the logarithmic gain for model complexity, I was still interested in testing model structures, leading me to use the model randomization within model_randomization.py to automate the testing of model frameworks. A few of these runs are attached in my train_results folder. (unfortunately, I am having git inconsistencies in most of my training data saves, I suspect due to the size of my commit, but there are still multiple runs stored within the directory). The randomization proved to perform relatively within the 80-90% accuracy range though, and I decided to take the training time into account by introducing an effectiveness score. Relatively ambiguous, this score provides a rough estimate of the model's accuracy in relation to training time. This evaluation should be tuned based on the use case, but as this project was exploratory, I didn't have any specific goal to fit.

Overall I learned a lot about the capabilities of neural networks and loved working on this project! I hope to utilize this to build a Lego sorter in the future!
