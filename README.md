# DBN-based NIDS on the CICIDS2017 Dataset
[![python](https://img.shields.io/badge/python-3.8.2-blue?style=plastic&logo=python)](https://www.python.org/downloads/release/python-382/)
[![pip](https://img.shields.io/badge/pypi-v22.1-informational?style=plastic&logo=pypi)](https://pypi.org/project/pip/22.1/)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/an-intrusion-detection-system-based-on-deep/network-intrusion-detection-on-cicids2017)](https://paperswithcode.com/sota/network-intrusion-detection-on-cicids2017?p=an-intrusion-detection-system-based-on-deep)

## Table of contents

* [Introduction](#introduction)
* [Installation](#installation)
    * [Dependencies](#dependencies)
* [Data Preparation](#data-preparation)
* [Usage](#usage)
* [Files and Folders structure](#files-and-folders-structure)
* [Requirements](#requirements)
* [References](#references)
* [License](#license)
* [Authors](#authors)


## Introduction

This is the source code for the paper entitled **["An Intrusion Detection System based on Deep Belief Networks"](https://arxiv.org/pdf/2207.02117.pdf)** presented in the **4th International Conference on Science of Cyber Security (SciSec)**.

In this repository, we propose a multi-class classification NIDS based on Deep Belief Networks (DBNs). DBN is a generative graphical model formed by stacking multiple Restricted Boltzmann Machines (RBMs). It can identify and learn high-dimensional representations due to its deep architecture. We conducted multiple experiments using the CICIDS2017 dataset [[1]](#1) with various class-balancing techniques.


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
* Afterwards, the dataset must be pre-processed using this following command line:
```bash
    $ make dataset
```
It will generate multiple pickle files that will we use to train and evaluate our models. More details about the pre-processing can be found [here](preprocessing/README.md#data-pre-processing-of-the-cicids2017).


## Usage

Once the data is ready to be used, you can train the models using configs files. Config files are in `.json` format:
```json
    {
        "name": "deep_belief_network",
        "model": {                                       
            "type": "DBN",
            "args": {                                       // model parameters
                "n_visible": 49,
                "n_hidden": [128, 256, 128, 128, 64],
                "n_classes": 6,
                "learning_rate": [0.1, 0.1, 0.1, 0.1, 0.1],
                "momentum": [0.9, 0.9, 0.9, 0.9, 0.9],
                "decay": [0, 0, 0, 0, 0],
                "batch_size": [64, 64, 64, 64, 64],
                "num_epochs": [10, 10, 10, 10, 10],
                "k": [1, 1, 1, 1, 1]
            }
        },
        "data_loader": {
            "type": "CICIDS2017DataLoader",                 // selecting data loader
            "args": {
                "batch_size": 128                           // batch size
            }
        },
        "optimizer": {
            "type": "Adam",
            "args": {
                "lr": 0.001,                                // learning rate
                "weight_decay": 0,                          // weight decay
                "amsgrad": false,
                "balanced": false
            }
        },
        "loss": {
            "type": "CrossEntropyLoss",                     // loss function
            "args": {
                "reduction": "mean"
            }
        },
        "trainer": {
            "num_epochs": 30                                // number of training epochs
        }
    }
```

Additional configurations can be added in the future, currently to start our DBN and MLP scripts please follow these simple commmands:
```bash
    # train the deep belief network
    $ python main.py --config ./configs/deepBeliefNetwork.json

    # train the multi-layer perceptron
    $ python main.py --config ./configs/multilayerPerceptron.json
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
    ├── LICENSE
    ├── main.py
    ├── Makefile
    ├── README.md                                           # top-level README for this project.
    └── requirements.txt                                    # requirements.txt file for reproducing the experiments.
```


## Requirements

All the experiments were conducted using a 64-bit Intel(R) Core(TM) i7-7500U CPU with 16GB RAM in Windows 10 environment. The models have been implemented in Python v3.8.2 using the PyTorch v1.9.0 library.


## References
<a id="1">[1]</a> 
Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, 
“Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”,
4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018


## License

This project is released under the [Apache 2.0 license](LICENSE).


## Authors

**Othmane Belarbi**, **Aftab Khan**, **Pietro Carnelli** and **Theodoros Spyridopoulos**,



## Citation

If you find this code useful in your research, please cite this article as:
```bibtex
@InProceedings{belarbi2022intrusion,
    author="Belarbi, Othmane and Khan, Aftab and Carnelli, Pietro and Spyridopoulos, Theodoros",
    editor="Su, Chunhua and Sakurai, Kouichi and Liu, Feng",
    title="An Intrusion Detection System Based on Deep Belief Networks",
    booktitle="Science of Cyber Security",
    year="2022",
    publisher="Springer International Publishing",
    address="Cham",
    pages="377--392",
    isbn="978-3-031-17551-0"
}
```
