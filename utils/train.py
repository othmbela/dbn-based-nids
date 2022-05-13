from tqdm import tqdm
import logging

import torch


def train(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    device: torch.device,
):
    """Train the network.

    Parameters
    ----------
    model: torch.nn.Module
        Neural network model used in this example.

    optimizer: torch.optim
        Optimizer.

    train_loader: torch.utils.data.DataLoader
        DataLoader used in training.

    valid_loader: torch.utils.data.DataLoader
        DataLoader used in validation.

    num_epochs: int
        Number of epochs to run in each round.

    device: torch.device
        (Default value = torch.device("cpu"))
        Device where the network will be trained within a client.

    Returns
    -------
        Tuple containing the history, the train_loss, the train_accuracy, the valid_loss, and the valid_accuracy.

    """

    model.to(device)

    history = {
        'train': {
            'total': 0,
            'loss': [],
            'accuracy': [],
            'output_pred': [],
            'output_true': []
        },
        'valid': {
            'total': 0,
            'loss': [],
            'accuracy': [],
            'output_pred': [],
            'output_true': []
        }
    }

    for epoch in range(1, num_epochs+1):

        ########################################
        ##             TRAIN LOOP             ##
        ########################################

        model.train()

        train_loss = 0.0
        train_steps = 0
        train_total = 0
        train_correct = 0

        train_output_pred = []
        train_output_true = []

        logging.info(f"Epoch {epoch}/{num_epochs}:")
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)

            # forward + backward + optimize
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
            train_steps += 1

            predicted = (torch.sigmoid(outputs) >= 0.21).float().squeeze()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_output_pred += predicted.tolist()
            train_output_true += labels.tolist()

        ########################################
        ##             VALID LOOP             ##
        ########################################
        model.eval()

        val_loss = 0.0
        val_steps = 0
        val_total = 0
        val_correct = 0

        val_output_pred = []
        val_output_true = []

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.cpu().numpy()
                val_steps += 1

                predicted = (torch.sigmoid(outputs) >= 0.21).float().squeeze()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_output_pred += predicted.tolist()
                val_output_true += labels.tolist()

        history['train']['total'] = train_total
        history['train']['loss'].append(train_loss/train_steps)
        history['train']['accuracy'].append(train_correct/train_total)
        history['train']['output_pred'] = train_output_pred
        history['train']['output_true'] = train_output_true

        history['valid']['total'] = val_total
        history['valid']['loss'].append(val_loss/val_steps)
        history['valid']['accuracy'].append(val_correct/val_total)
        history['valid']['output_pred'] = val_output_pred
        history['valid']['output_true'] = val_output_true

        logging.info(f'loss: {train_loss/train_steps} - acc: {train_correct/train_total} - val_loss: {val_loss/val_steps} - val_acc: {val_correct/val_total}')

    logging.info(f"Finished Training")

    return history
