from tqdm import tqdm
import logging

import torch


def test(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    """Validate the network.

    Parameters
    ----------
    model: torch.nn.ModuleList
        Neural network model used in this example.

    test_loader: torch.utils.data.DataLoader
        DataLoader used in testing.

    device: torch.device
        (Default value = torch.device("cpu"))
        Device where the network will be trained within a client.

    Returns
    -------
        Tuple containing the history, and a detailed report.

    """

    model.eval()

    history = {
        'test': {
            'total': 0,
            'loss': 0.0,
            'accuracy': 0.0,
            'output_pred': [],
            'output_true': [],
            'output_pred_prob': []
        }
    }

    test_loss = 0.0
    test_steps = 0
    test_total = 0
    test_correct = 0

    test_output_pred = []
    test_output_true = []
    test_output_pred_prob = []

    with torch.no_grad():
        for (inputs, labels) in tqdm(test_loader):

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            
            loss = criterion(outputs, labels.unsqueeze(1))
            test_loss += loss.cpu().item()
            test_steps += 1

            predicted = (torch.sigmoid(outputs) >= 0.21).float().squeeze()
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            test_output_pred += predicted.tolist()
            test_output_true += labels.tolist()
            test_output_pred_prob += torch.sigmoid(outputs).squeeze().tolist()

    history['test']['total'] = test_total
    history['test']['loss'] = test_loss/test_steps
    history['test']['accuracy'] = test_correct/test_total
    history['test']['output_pred'] = test_output_pred
    history['test']['output_true'] = test_output_true
    history['test']['output_pred_prob'] = test_output_pred_prob

    logging.info(f'Test loss: {test_loss/test_steps}, Test accuracy: {test_correct/test_total}')

    return history
