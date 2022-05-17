from sklearn.metrics import classification_report, f1_score
import argparse
import logging
import os

import torch
import torch.optim as optim

from logger import setup_logging
from utils import (
    datasets,
    models,
    test,
    train,
    utils,
    visualisation,
)


LOG_CONFIG_PATH = os.path.join(os.path.abspath("."), "logger", "logger_config.json")
LOG_DIR   = os.path.join(os.path.abspath("."), "logs")
DATA_DIR  = os.path.join(os.path.abspath('.'), "data", "processed")
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
    train_loader, valid_loader, test_loader = datasets.load_data(
        data_path=DATA_DIR,
        batch_size=config["data_loader"]["args"]["batch_size"]
    )
    logging.info("Dataset loaded!")

    logging.info("Start training the model...")
    optimizer = getattr(torch.optim, config["optimizer"]["type"])(params=model.parameters(), **config["optimizer"]["args"])
    criterion = getattr(torch.nn, config["loss"]["type"])(**config["loss"]["args"])

    _ = train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=config["trainer"]["num_epochs"],
        device=DEVICE
    )
    logging.info(f'{config["name"]} model trained!')

    labels = ['Benign', 'Botnet ARES', 'Brute Force', 'DoS/DDoS', 'PortScan', 'Web Attack']

    logging.info("Training Set -- Classification Report", end="\n\n")
    logging.info(classification_report(
        y_true=history['train']['output_true'],
        y_pred=history['train']['output_pred'],
        target_names=labels
    ))

    logging.info("Validation Set -- Classification Report", end="\n\n")
    logging.info(classification_report(
        y_true=history['validation']['output_true'],
        y_pred=history['validation']['output_pred'],
        target_names=labels
    ))

    logging.info(f'Evaluate {config["name"]} model')
    testing_history = test(
        model=model,
        criterion=criterion,
        test_loader=test_loader,
        device=DEVICE
    )

    y_true = testing_history["test"]["output_true"]
    y_pred = testing_history["test"]["output_pred"]

    logging.info(f'Classification Report {config["name"]}\n')
    logging.info(classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=labels
    ))

    utils.mkdir(IMAGE_DIR)
    visualisation.plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        save=True,
        save_dir=IMAGE_DIR,
        filename=f'{config["name"]}_test_confusion_matrix.png'
    )

    path = os.path.join(MODEL_DIR, f'{config["name"]}.pt')
    utils.mkdir(MODEL_DIR)
    torch.save({
        'epoch': config["trainer"]["num_epochs"],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
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
