from tqdm import tqdm
import logging

import torch
import torch.nn as nn

from models.RBM import RBM


class DBN(nn.Module):
    """This class implements a Deep Belief Networks."""

    def __init__(self,
                 n_visible=49,
                 n_hidden=(128, 128, 64),
                 n_classes=6,
                 learning_rate=(0.1, 0.1, 0.1),
                 momentum=(0.9, 0.9, 0.9),
                 decay=(0, 0, 0),
                 batch_size=(64, 64, 64),
                 num_epochs=(10, 10, 10),
                 k=(1, 1, 1),
                 device="cpu"):
        """Initialization method.

        Parameters
        ----------
            n_visible (int): Amount of visible units.
            n_hidden (tuple): Amount of hidden units per layer.
            n_classes (int): Number of classes.
            learning_rate (tuple): Learning rate per layer.
            n_classes (int): Number of classes.
            momentum (tuple): Momentum parameter per layer.
            decay (tuple): Weight decay used for penalization per layer.
            batch_size (tuple): Batch size per layer.
            num_epochs (tuple): Number of epochs per layer.
            k (tuple): Number of Gibbs' sampling k per layer.
        """

        super(DBN, self).__init__()

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = len(n_hidden)
        self.lr = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.k = k
        self.models = []

        # For every possible layer
        for i in range(self.n_layers):
            if i == 0:
                rbm = RBM(self.n_visible,
                          self.n_hidden[i],
                          learning_rate=self.lr[i],
                          momentum=self.momentum[i],
                          decay=self.decay[i],
                          batch_size=self.batch_size[i],
                          num_epochs=self.num_epochs[i],
                          k=self.k[i],
                          device=device)
            else:
                rbm = RBM(self.n_hidden[i-1],
                          self.n_hidden[i],
                          learning_rate=self.lr[i],
                          momentum=self.momentum[i],
                          decay=self.decay[i],
                          batch_size=self.batch_size[i],
                          num_epochs=self.num_epochs[i],
                          k=self.k[i],
                          device=device)

            # Appends the model to the list
            self.models.append(rbm)

        # Creating the Fully Connected layer to append on top of DBNs
        self.fc = nn.Linear(self.n_hidden[-1], self.n_classes)

    def reconstruct(self, data_loader):
        """Reconstructs batches of new input_data.

        Parameters
        ----------
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.

        Returns
        -------
            Reconstruction error and visible probabilities, i.e., P(v|h).
        """

        # Resetting MSE to zero
        mse = 0

        # Defining the batch size as the amount of input_data in the dataset
        batch_size = len(data_loader.dataset)

        # For every batch
        for input_data, _ in tqdm(data_loader):

            # Applying the initial hidden probabilities as the input_data
            hidden_probs = input_data

            # For every possible model (RBM)
            for model in self.models:
                # Flattening the hidden probabilities
                hidden_probs = hidden_probs.reshape(
                    batch_size, model.n_visible)

                # Performing a hidden layer sampling
                hidden_probs, _ = model.sample_hidden(hidden_probs)

            # Applying the initial visible probabilities as the hidden probabilities
            visible_probs = hidden_probs

            # For every possible model (RBM)
            for model in reversed(self.models):
                # Flattening the visible probabilities
                visible_probs = visible_probs.reshape(
                    batch_size, model.n_hidden)

                # Performing a visible layer sampling
                visible_probs, visible_states = model.sample_visible(
                    visible_probs)

            # Calculating current's batch reconstruction MSE
            batch_mse = torch.div(
                torch.sum(torch.pow(input_data - visible_states, 2)), batch_size)

            # Summing up to reconstruction's MSE
            mse += batch_mse

        # Normalizing the MSE with the number of batches
        mse /= len(data_loader)

        return mse, visible_probs

    def forward(self, x):
        """Performs a forward pass over the data.

        Parameters
        ----------
            x (torch.Tensor): An input tensor for computing the forward pass.

        Returns
        -------
            A tensor containing the DBN's outputs.
        """

        # For every possible model, calculates the outputs of current model
        for model in self.models:
            x = model(x)

        # Calculating the fully-connected outputs
        out = self.fc(x)
        return out

    def fit(self, train_loader):
        """Fits a new DBN model.

        Parameters
        ----------
            dataset (torch.utils.data.Dataset | Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of input_data per batch.
            epochs (tuple): Number of training epochs per layer.

        Returns
        -------
            MSE (mean squared error) and log pseudo-likelihood from the training step.
        """

        # Initializing MSE and pseudo-likelihood as lists
        mse, pl = [], []

        # Initializing the dataset's variables
        input_data_loader = train_loader
        input_data = torch.tensor(train_loader.dataset.features.values)

        # For every possible model (RBM)
        for i, model in enumerate(self.models):
            logging.info(f'Fitting layer {i+1}/{self.n_layers}')

            # Fits the RBM
            model_mse, model_pl = model.fit(input_data_loader)

            # Performs a forward pass over the input_data to get their probabilities
            input_data, _ = model.sample_hidden(input_data.float())

            # Detaches the variable from the computing graph
            input_data = input_data.detach()

            # Create new dataloader
            tensor_x = input_data.type(torch.FloatTensor)
            tensor_y = input_data.type(torch.FloatTensor)

            dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
            input_data_loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size[i],
                shuffle=True
            )

            # Appending the metrics
            mse.append(model_mse)
            pl.append(model_pl)

        return mse, pl