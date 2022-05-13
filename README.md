# ML-based IDS on the CICIDS2017 Dataset
[![python](https://img.shields.io/badge/python-3.8.2-blue?style=plastic&logo=python)](https://www.python.org/downloads/release/python-382/)
[![pip](https://img.shields.io/badge/pypi-v21.3.1-informational?style=plastic&logo=pypi)](https://pypi.org/project/pip/21.3.1/)

[[_TOC_]]

## Introduction

Several Intrusion Detection System (IDS) has been developed. Traditional IDS base their operation on Machine Learning models learned centrally in the cloud and then distributed across multiple hosts. However, this centraliSed approach hinders knowledge sharing among system owners due to privacy/secrecy violation concerns. 

![Network Intrusion Detection System](./assets/network-intrusion-detection.png)


## Project Setup

Start by cloning the project:
```bash
    $ git clone git@ssh.dev.azure.com:v3/toshiba-bril/BRIL%20Federated%20Learning/DBN-FL
```

Then cd into your the cloned repository as such:
```bash
    $ cd ml-based-ids
```

Project dependencies (such as `torch` and `flwr`) are defined in `requirements.txt`. You can install those dependencies and manage your dependencies using Python virtual environment, but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

1. Create and fire up your virtual environment:
    ```bash
        $ python3 -m venv venv
        $ source venv/bin/activate
    ```
2. Install the dependencies needed to run the app:
    ```bash
        $ pip install -r requirements.txt
    ```


## Data Preparation

* Download the dataset from the Kaggle competition [here](https://www.kaggle.com/c/instacart-market-basket-analysis/overview).
* Move the CSV files to the following directory ***./data/raw/***
* Then, create and preprocess the dataset using this following command line:
```bash
    $ make dataset
```
It will generate multiple pickle files that will we use to train our models.


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
    │   ├── interim                                         # intermediate data that has been transformed.
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
    │   ├── NN.py
    │   └── RNN.py
    │
    ├── notebooks/                                          # jupyter notebooks.
    │   └── XGBoost Classifier.ipynb                        # benchmark implementation.
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

## Author

**Othmane Belarbi**
