from tqdm import tqdm
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class RBM(nn.Module):
    """This class implements a Binary Restricted Boltzmann Machine."""

    def __init__(self,
                 n_visible=49,
                 n_hidden=128,
                 learning_rate=0.1,
                 momentum=0.9,
                 decay=0,
                 batch_size=64,
                 num_epochs=10,
                 k=1,
                 device="cpu"):

        super(RBM, self).__init__()

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.k = k
        self.device = device

        # Initialize weights and biases
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.vb = nn.Parameter(torch.zeros(self.n_visible))
        self.hb = nn.Parameter(torch.zeros(self.n_hidden))

        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)

    def forward(self, x):
        """Performs a forward pass over the data.

        Parameters
        ----------
            x (torch.Tensor): An input tensor for computing the forward pass.

        Returns
        -------
            A tensor containing the RBM's outputs.

        """

        # Calculates the outputs of the model
        x, _ = self.sample_hidden(x)

        return x

    def backward(self, h):
        """Reconstruct visible units given the hidden layer output.

        Parameters
        ----------
            transformed_data: array-like, shape = (n_samples, n_features)

        Returns
        -------
        """
        return self.sample_visible(h)

    def sample_hidden(self, v):
        '''Sample from the distribution P(h|v).

        Parameters
        ----------
            v : ndarray of shape (n_samples, n_features)
                Values of the visible layer to sample from.
        '''

        activations = F.linear(v, self.W.t(), self.hb)
        p_h_given_v = torch.sigmoid(activations)

        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_visible(self, h):
        '''Sample from the distribution P(v|h).

        Parameters
        ----------
            h : ndarray of shape (n_samples, n_components)
                Values of the hidden layer to sample from.
        '''
        activations = F.linear(h, self.W, self.vb)
        p_v_given_h = torch.sigmoid(activations)

        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def gibbs_sampling(self, v):
        """Performs the whole Gibbs sampling procedure.

        Parameters
        ----------
            v (torch.Tensor): A tensor incoming from the visible layer.

        Returns
        -------
            The probabilities and states of the hidden layer sampling (positive),
            the probabilities and states of the hidden layer sampling (negative)
            and the states of the visible layer sampling (negative).

        """

        # Calculating positive phase hidden probabilities and states
        pos_hidden_probs, pos_hidden_states = self.sample_hidden(v)

        # Initially defining the negative phase
        neg_hidden_states = pos_hidden_states

        # Performing the Contrastive Divergence
        for _ in range(self.k):
            # Calculating visible probabilities and states
            _, visible_states = self.sample_visible(neg_hidden_states)

            # Calculating hidden probabilities and states
            neg_hidden_probs, neg_hidden_states = self.sample_hidden(
                visible_states
            )

        return pos_hidden_probs, pos_hidden_states, neg_hidden_probs, neg_hidden_states, visible_states

    def energy(self, samples):
        """Calculates and frees the system's energy.

        Parameters
        ----------
            samples (torch.Tensor): Samples to be energy-freed.

        Returns
        -------
            The system's energy based on input samples.

        """

        # Calculate samples' activations
        activations = F.linear(samples, self.W.t(), self.hb)

        # Creating a Softplus function for numerical stability
        s = nn.Softplus()

        # Calculate the hidden term
        h = torch.sum(s(activations), dim=1)

        # Calculate the visible term
        v = torch.mv(samples, self.vb)

        # Finally, gathers the system's energy
        energy = -v - h

        return energy

    def pseudo_likelihood(self, samples):
        """Calculates the logarithm of the pseudo-likelihood.

        Parameters
        ----------
            samples (torch.Tensor): Samples to be calculated.

        Returns
        -------
            The logarithm of the pseudo-likelihood based on input samples.

        """

        # Gathering a new array to hold the rounded samples
        samples_binary = torch.round(samples)

        # Calculates the energy of samples before flipping the bits
        energy = self.energy(samples_binary)

        # Samples an array of indexes to flip the bits
        indexes = torch.randint(0, self.n_visible, size=(
            samples.size(0), 1), device=self.device)

        # Creates an empty vector for filling the indexes
        bits = torch.zeros(samples.size(
            0), samples.size(1), device=self.device)

        # Fills the sampled indexes with 1
        bits = bits.scatter_(1, indexes, 1)

        # Actually flips the bits
        samples_binary = torch.where(
            bits == 0, samples_binary, 1 - samples_binary)

        # Calculates the energy after flipping the bits
        energy1 = self.energy(samples_binary)

        # Calculate the logarithm of the pseudo-likelihood
        pl = torch.mean(self.n_visible * torch.log(torch.sigmoid(energy1 - energy) + 1e-10))

        return pl

    def fit(self, train_loader):
        """Fits a new RBM model.

        Parameters
        ----------
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (int): Number of training epochs.

        Returns
        -------
            MSE (mean squared error) and log pseudo-likelihood from the training step.

        """

        # For every epoch
        for epoch in range(1, self.num_epochs+1):

            # Resetting epoch's MSE and pseudo-likelihood to zero
            mse, pl = 0, 0

            # For every batch
            for inputs, _ in tqdm(train_loader):

                inputs = inputs.to(self.device)

                # Performs the Gibbs sampling procedure
                _, _, _, _, visible_states = self.gibbs_sampling(
                    inputs.float()
                )

                # Detaching the visible states from GPU for further computation
                visible_states = visible_states.detach()

                # Calculates the loss for further gradients' computation
                cost = torch.mean(self.energy(inputs.float())) - torch.mean(self.energy(visible_states))

                # Initializing the gradient
                self.optimizer.zero_grad()

                # Computing the gradients
                cost.backward()

                # Updating the parameters
                self.optimizer.step()

                # Gathering the size of the batch
                batch_size = inputs.size(0)

                # Calculating current's batch MSE
                batch_mse = torch.div(
                    torch.sum(torch.pow(inputs.float() - visible_states, 2)), batch_size).detach()

                # Calculating the current's batch logarithm pseudo-likelihood
                batch_pl = self.pseudo_likelihood(inputs.float()).detach()

                # Summing up to epochs' MSE and pseudo-likelihood
                mse += batch_mse
                pl += batch_pl

            # Normalizing the MSE and pseudo-likelihood with the number of train_loader
            mse /= len(train_loader)
            pl /= len(train_loader)

            logging.info(
                f"Epoch {epoch}/{self.num_epochs} - MSE: {mse} - Pseudo-Likelihood: {pl}")

        return mse, pl
