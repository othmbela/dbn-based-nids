from sklearn.metrics import classification_report, f1_score
import pandas as pd
import numpy as np
import argparse
import logging
import os

import torch
import torch.optim as optim

from logger import setup_logging
from utils import (
    dataset,
    models,
    test,
    train,
    utils,
    visualisation,
)


LOG_CONFIG_PATH = os.path.join(os.path.abspath("."), "logger", "logger_config.json")
LOG_DIR   = os.path.join(os.path.abspath("."), "logs")
DATA_DIR  = os.path.join(os.path.abspath('.'), "data")
IMAGE_DIR = os.path.join(os.path.abspath("."), "images")
MODEL_DIR = os.path.join(os.path.abspath("."), "checkpoints")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure that all operations are deterministic for reproducibility, even on GPU (if used)
utils.set_seed(42)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

def main(config):
    """Centralised"""

    # Configure logging module
    utils.mkdir(LOG_DIR)
    setup_logging(save_dir=LOG_DIR, log_config=LOG_CONFIG_PATH)

    logging.info(f'######## Training the {config["name"]} model ########')
    model = models.load_model(model_name=config["model"]["type"], params=config["model"]["args"])
    model.to(DEVICE)

    logging.info("Loading dataset...")
    train_loader, valid_loader, test_loader = dataset.load_data(
        data_path=DATA_DIR,
        balanced=config["data_loader"]["args"]["balanced"],
        batch_size=config["data_loader"]["args"]["batch_size"],
    )
    logging.info("Dataset loaded!")

    # TODO: Optimiser as an array
    logging.info("Start training the model...")
    optimizer = [getattr(torch.optim, config["optimizer"]["type"])(params=model.parameters(), **config["optimizer"]["args"])]
    criterion = getattr(torch.nn, config["loss"]["type"])(**config["loss"]["args"])

    train_history = train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=config["trainer"]["num_epochs"],
        device=DEVICE
    )
    logging.info(f'{config["name"]} model trained!')

    train_output_true = train_history["train"]["output_true"]
    train_output_pred = train_history["train"]["output_pred"]
    valid_output_true = train_history["valid"]["output_true"]
    valid_output_pred = train_history["valid"]["output_pred"]

    labels = ["Benign", "Botnet ARES", "Brute Force", "DoS/DDoS", "PortScan", "Web Attack"]

    ## Training Set results
    logging.info('Training Set -- Classification Report')
    logging.info(classification_report(
        y_true=train_output_true,
        y_pred=train_output_pred,
        target_names=labels
    ))
    
    visualisation.plot_confusion_matrix(
        y_true=train_output_true,
        y_pred=train_output_pred,
        labels=labels,
        save=True,
        save_dir=IMAGE_DIR,
        filename=f'{config["name"]}_train_confusion_matrix.pdf'
    )

    ## Validation Set results
    logging.info('Validation Set -- Classification Report')
    logging.info(classification_report(
        y_true=valid_output_true,
        y_pred=valid_output_pred,
        target_names=labels
    ))

    visualisation.plot_confusion_matrix(
        y_true=valid_output_true,
        y_pred=valid_output_pred,
        labels=labels,
        save=True,
        save_dir=IMAGE_DIR,
        filename=f'{config["name"]}_train_confusion_matrix.pdf'
    )

    logging.info(f'Evaluate {config["name"]} model')
    test_history = test(
        model=model,
        criterion=criterion,
        test_loader=test_loader,
        device=DEVICE
    )

    test_output_true = test_history["test"]["output_true"]
    test_output_pred = test_history["test"]["output_pred"]
    test_output_pred_prob = test_history["test"]["output_pred_prob"]

    ## Testing Set results
    logging.info(f'Testing Set -- Classification Report {config["name"]}\n')
    logging.info(classification_report(
        y_true=test_output_true,
        y_pred=test_output_pred,
        target_names=labels
    ))

    utils.mkdir(IMAGE_DIR)
    visualisation.plot_confusion_matrix(
        y_true=test_output_true,
        y_pred=test_output_pred,
        labels=labels,
        save=True,
        save_dir=IMAGE_DIR,
        filename=f'{config["name"]}_test_confusion_matrix.pdf'
    )

    y_test = pd.get_dummies(test_output_true).values
    y_score = np.array(test_output_pred_prob)

    # Plot ROC curve
    visualisation.plot_roc_curve(
        y_test=y_test,
        y_score=y_score,
        labels=labels,
        save=True,
        save_dir=IMAGE_DIR,
        filename=f'{config["name"]}_roc_curve.pdf'
    )

    # Plot Precision vs. Recall curve
    visualisation.plot_precision_recall_curve(
        y_test=y_test,
        y_score=y_score,
        labels=labels,
        save=True,
        save_dir=IMAGE_DIR,
        filename=f'{config["name"]}_prec_recall_curve.pdf'
    )

    path = os.path.join(MODEL_DIR, f'{config["name"]}.pt')
    utils.mkdir(MODEL_DIR)
    torch.save({
        'epoch': config["trainer"]["num_epochs"],
        'model_state_dict': model.state_dict(),
    }, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        required=True,
        help="Config file path. (default: None)"
    )
    args = parser.parse_args()

    config = utils.read_json(args.config)
    main(config)
