from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import pickle
import json
import os

import torch


def mkdir(dir: str):
    """Similar to "mkdir" in bash.
    
    Create a directory with path 'dir' if it does not exist.
    """
    if not os.path.exists(dir):
        logging.debug(f"The following path {dir} doesn't exist.")
        os.makedirs(dir)
        logging.debug(f"{dir} successfully created.")


def set_seed(seed: int):
    """Function for setting the seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_samples_weight(target):
    """Get Samples Weight"""

    class_sample_count = np.bincount(target)
    weight = 1. / class_sample_count

    samples_weight = torch.tensor([weight[t] for t in target])
    samples_weight = samples_weight.double()

    return samples_weight


def write_json(content, filename):
    with open(filename, 'w') as write_file:
        json.dump(content, write_file)


def read_json(filename):
    with open(filename, 'r') as read_file:
        return json.load(read_file)


def write_pickle(content, filename):
    with open(filename, "wb") as write_file:
        pickle.dump(content, write_file)


def read_pickle(filename):
    with open(filename, 'rb') as read_file:
        return pickle.load(read_file)
