# DBN-based NIDS on the CICIDS2017 Dataset
[![python](https://img.shields.io/badge/python-3.8.2-blue?style=plastic&logo=python)](https://www.python.org/downloads/release/python-382/)
[![pip](https://img.shields.io/badge/pypi-v22.1-informational?style=plastic&logo=pypi)](https://pypi.org/project/pip/22.1/)


## Table of contents

* [Introduction](#introduction)
* [Installation](#installation)
    * [Dependencies](#dependencies)
* [Data Preparation](#data-preparation)
* [Usage](#usage)
* [Files and Folders structure](#files-and-folders-structure)
* [Requirements](#requirements)
* [Authors](#authors)


## Introduction

In this repository, we propose a multi-class classification NIDS based on Deep Belief Networks (DBNs). DBN is a generative graphical model formed by stacking multiple Restricted Boltzmann Machines (RBMs). It can identify and learn high-dimensional representations due to its deep architecture. We conducted multiple experiments using the CICIDS2017 dataset with various class-balancing techniques.


## Installation

* If you want to run the scripts, first ensure you have python globally installed in your computer. If not, you can get python [here](https://www.python.org).
* Then, clone the repo to your PC and change the branch:
    ```bash
        $ git clone https://github.com/othmbela/dbn-based-nids.git
    ```

* ### Dependencies
    1. Cd into your the cloned repository as such:
        ```bash
            $ cd dbn-based-nids
        ```
    2. Initialise the project as such:
        ```bash
            $ make init
        ```
    First, the command line will create your virtual environment and install the dependencies needed to run the app. Then, it will create the data folders.


## Data Preparation

* Download the dataset from [here](https://www.unb.ca/cic/datasets/ids-2017.html).
* Move the CSV files to the following directory ***./data/raw/***
* Then, create and pre-process the dataset using this following command line:
```bash
    $ make dataset
```
It will generate multiple pickle files that will we use to train and evaluate our models. More details about the pre-processing can be found [here](preprocessing/README.md#data-pre-processing-of-the-cicids2017).


## Usage

Once the data is ready to be used, you can train the models using configs files. Config files are in `.json` format:
```json
    {
        "name": "recurrent_neural_network",                 // model name
        "model": {
            "type": "RNN",
            "args": {                                       // model parameters
                "input_size": 61,
                "sequence_length": 1,
                "hidden_size": 64,
                "num_layers": 3,
                "nonlinearity": "tanh",
                "dropout": 0.3,
                "output_size": 1
            }
        },
        "data_loader": {
            "type": "InstacartDataLoader",                  // selecting data loader
            "args": {
                "batch_size": 128                           // batch size
            }
        },
        "optimizer": {
            "type": "Adam",
            "args": {
                "lr": 0.001,                                // learning rate
                "weight_decay": 0,                          // weight decay
                "amsgrad": false
            }
        },
        "loss": {
            "type": "BCEWithLogitsLoss",                    // loss function
            "args": {
                "reduction": "mean"
            }
        },
        "lr_scheduler": {
            "type": "StepLR",                               // learning rate scheduler
            "args": {
                "step_size": 1,
                "gamma": 0.95
            }
        },
        "trainer": {
            "num_epochs": 5                                // number of training epochs
        }
    }
```

Additional configurations can be added in the future, currently to start our RNN and NN scripts please follow these simple commmands:
```bash
    # train the recurrent neural network
    $ python main.py --config ./configs/recurrentNeuralNetwork.json

    # train the neural network
    $ python main.py --config ./configs/neuralNetwork.json
```

## Files and Folders structure

```
    ├── checkpoints/                                        # store the trained models as *.pt file.
    │
    ├── configs/
    │
    ├── data/                                               # default directory for storing input data.
    │   ├── processed                                       # final data for modelling.
    │   └── raw                                             # original data.
    │
    ├── images/                                             # store images
    │
    ├── logger/                                             # setup the logger using logger_config.json
    ├── logs/                                               # store *.logs
    │
    ├── models/                                             # pytorch models.
    │   ├── __init__.py
    │   ├── DBN.py
    │   ├── MLP.py
    │   └── RBM.py
    │
    ├── notebooks/                                          # jupyter notebooks.
    │
    ├── preprocessing/                                      # scripts for preprocessing the dataset.
    │
    ├── utils/
    │   ├── __init__.py
    │   ├── datasets.py
    │   ├── models.py
    │   ├── test.py                                         # evaluation of trained model.
    │   ├── train.py                                        # main script to start training.
    │   ├── utils.py                                        # small utility functions.
    │   └── visualisation.py                                # functions to visualise the results.
    │
    ├── venv/                                               # virtual environment.
    │
    ├── .gitignore
    ├── main.py
    ├── Makefile
    ├── README.md                                           # top-level README for this project.
    └── requirements.txt                                    # requirements.txt file for reproducing the experiments.
```

## Requirements

All the experiments were conducted using a 64-bit Intel(R) Core(TM) i7-7500U CPU with 16GB RAM in a Ubuntu-Linux environment. The models have been implemented and trained in Python v3.8.2 using the Pytorch v1.10.2 library.

## Authors

**Othmane Belarbi**, **Theo Spyridopoulos**, **Aftab Khan** and **Pietro Carnelli**
