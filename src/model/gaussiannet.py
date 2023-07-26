import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Independent, Normal

from .modules import DoubleConv, Down


class GaussianNet(nn.Module):
    """
    Axis-aligned covariance gaussian distribution encoder network. It encodes
    the images (prior) or images and labels (posterior) into a mean and
    logvariance like a classic VAE, which are then used to model an axis-aligned
    covariance multivariate gaussian distribution.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Works for both RGB and grayscale images.
    n_classes : int
        Number of output classes. Refers number of labels in the segmentation
        plus the background (eg. 4 labels + background -> nclasses = 5).
    latent_dim : int
        Dimension of the latent space.
    n_layers : int
        Number of layers for the encoder architecture. Defaults to 4, keeping
        number of layers as the regular U-Net encoder.
    filter_factor : float
        Factor to adjust the number of filters in each layer. Defaults to 1,
        with initial number of filters being 64 as the regular U-Net.
    posterior : bool
        Choose whether to use the network to generate a posterior distribution,
        making it a conditional VAE (cVAE). Defaults to False.
    kohl : bool
        Compute the mean and the variance using Kohl et. al's original method.
        If False uses linear layers like traditional VAEs. Defaults to True.
    linear_dim : int
        Input dimension of the linear layers if kohl=False. It depends on the
        input size and the parameters of the linear dimensions, and it is
        computed as W*H for 2D images and W*H*D for 3D images, being W, H and D
        the width, height and depth of the output of the last conv layer before
        the linear layer. Defauts to None.
    is3d : bool
        True if input images are 3D. Defaults to False.

    Attributes
    ----------
    self.filters : list
        List containing the number of filters used per layer. The length of the
        list is the exact same as n_layers.
    self.posterior : bool
        Same as parameter.
    self.latent_dim : bool
        Same as parameter.
    self.kohl : bool
        Same as parameter.
    self.is3d : bool
        Same as parameter.
    self.in_channels : int
        Number of input channels. If posterior=True, then the number of
        channels is equal to the image input channels plus the input channels
        of the labels/mask since the input would be a concatenation of both.
    self.encoder : nn.Sequential
        Encoder module responsible for downsampling the input tensor.
    self.conv_layer : torch.nn.Module, optional
        Convolutional layer used for Kohl's method of computing the mean and
        logarithmic variance. Only exists if kohl=True.
    self.fc_mu : torch.nn.Module, optional
        Linear layer that computes the encoded mean. Only exists if kohl=False.
    self.fc_logvar : torch.nn.Module, optional
        Linear layer that computes the encoded logarithmic variance. Only
        exists if kohl=False.

    Methods
    -------
    _create_encoder()
        Function used to create the encoder path of the U-Net
    _compute_mu_logvar(x)
        Computes the mean and logarithmic variance. Uses either Khol's or
        traditional method using linear layers.
    forward(x, label)
        Forward pass of the network. If posterior, label must not be None.
    """

    def __init__(self, in_channels, n_classes, latent_dim, n_layers=4,
                 filter_factor=1, posterior=False, kohl=True, linear_dim=None,
                 is3d=False):

        super(GaussianNet, self).__init__()

        # Calculate the number of filters for each layer based on the factor
        base_filters = 64 // filter_factor
        self.filters = [base_filters * (2 ** i) for i in range(n_layers)]
        self.posterior = posterior
        self.latent_dim = latent_dim
        self.kohl = kohl
        self.is3d = is3d

        # Define the input channels
        if self.posterior is True:
            # n_classes channels added, one per label
            self.in_channels = in_channels + n_classes
        else:
            self.in_channels = in_channels

        # Create the encoder
        self.encoder = self._create_encoder()

        # Define a convolutional layer for Kohl's method
        if self.kohl is True:
            if self.is3d is True:
                self.conv_layer = nn.Conv3d(in_channels=self.filters[-1],
                                            out_channels=2 * self.latent_dim,
                                            kernel_size=1,
                                            stride=1)
            else:
                self.conv_layer = nn.Conv2d(in_channels=self.filters[-1],
                                            out_channels=2 * self.latent_dim,
                                            kernel_size=1,
                                            stride=1)

        # Else create the mean and logvar linear layers for the classic method
        else:
            self.fc_mu = nn.Linear(self.filters[-1] * linear_dim, latent_dim)
            self.fc_logvar = nn.Linear(self.filters[-1] * linear_dim, latent_dim)

    def _create_encoder(self):
        """
        Function used to create the encoder network It takes a list
        containing the number of features, iterates through it and appends each
        encoder layer to an empty list. Then, the list is used to create a
        torch.nn.Sequential object that can be used in the forward pass.

        Returns
        -------
        encoder : torch.nn.Sequential
            Encoder path of the U-Net.
        """

        # Create empty list to store each encoder layer
        encoder_layers = []
        # First decoder layer takes the input channels, so create it separately
        encoder_layers.append(DoubleConv(
            self.in_channels, self.filters[0],
            dropout_p=0, is3d=self.is3d, bn=False))
        # Create the rest of the layers using a for loop
        for i in range(len(self.filters) - 1):
            encoder_layers.append(
                Down(self.filters[i], self.filters[i + 1],
                     dropout_p=0, is3d=self.is3d, bn=False))

        encoder = nn.Sequential(*encoder_layers)

        return encoder

    def _compute_mu_logvar(self, x):
        """
        Compute the mean and logarithmic variance of the input tensor 'x' to
        generate the latent variables. If `self.kohl` is True, the following
        operations are performed:

        1.  Compute the mean across spatial dimensions, yielding a vector of
            shape [B, C, 1, 1, *1], where B represents the batch size and C
            represents the number of channels.
        2.  Apply a convolution operation to reduce the dimensionality of the
            mean tensor to shape [B, 2 * self.latent_dim, 1, 1, *1].
        3.  Squeeze the spatial dimensions of the mean tensor to obtain a vector
            of shape [B, 2 * self.latent_dim].
        4.  Extract the first self.latent_dim elements from the mean tensor,
            representing the mean values.
        5.  Extract the remaining self.latent_dim elements from the mean tensor,
            representing the logarithmic variances.

        Else, it just reduces the dimensions of the encoded input tensor 'x'
        into the latent space dimension using a linear layer to produce the
        mean and logarithmic variance.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, C, *D, H, W], where B represents the
            batch size, C represents the number of channels, and H, W, *D
            represent the spatial dimensions of the input image.

        Returns
        -------
        mu : torch.Tensor
            Mean tensor of shape [B, self.latent_dim], representing the mean
            values of the latent variables.
        logvar : torch.Tensor
            Logarithmic variance tensor of shape [B, self.latent_dim],
            representing the logarithmic variances of the latent variables.
        """

        if self.kohl:
            # Determine the spatial dimensions
            spatial_dims = [2, 3] if len(x.shape) == 4 else [2, 3, 4]

            # Compute the mean across spatial dimensions
            encoding = torch.mean(x, dim=spatial_dims, keepdim=True)

            # Apply a convolution to reduce dimensionality
            mu_logvar = self.conv_layer(encoding)

            # Squeeze the spatial dimensions
            mu_logvar = torch.squeeze(mu_logvar, dim=spatial_dims)

            # Extract the mean and logarithmic variance
            mu = mu_logvar[:, :self.latent_dim]
            logvar = mu_logvar[:, self.latent_dim:]

        else:
            # Flatten the features
            x = x.view(x.size(0), -1)

            # Compute the mean and logvar of the latent space
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)

        return mu, logvar

    def forward(self, x, label=None, return_params=False):
        """
        Forward pass of the GaussianNet. Applied the encoding process and the
        computation of the mean and logarithmic variance, which are then used
        to generate an axis-aligned covariance multivariate gaussian proability
        distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, C, *D, H, W], where B represents the
            batch size, C represents the number of channels, and H, W, *D
            represent the spatial dimensions of the input image.
        label : torch.Tensor
            Mask/label tensor of shape [B, L, *D, H, W], where B represents the
            batch size, L represents the number of labels, and H, W, *D
            represent the spatial dimensions of the input mask. Only used when
            self.posterior=True. Defaults to None.

        Returns
        -------
        gaussian : torch.distributions.Distribution
            Axis-aligned covariance Gaussian distribution.
        """

        # Concatenate image and mask if the net is posterior (cVAE)
        if self.posterior:
            assert label is not None, 'Can not build posterior as label tensor is empty'
            x = torch.cat([x, label], dim=1)

        # Encoder path
        for encoder in self.encoder:
            x = encoder(x)

        # Compute the mean and the logarithmic variance
        mu, logvar = self._compute_mu_logvar(x)

        if return_params is True:
            return mu, logvar

        else:
            # Compute the standard deviation
            std = torch.exp(0.5 * logvar)

            # Generate an axis-aligned (diagonal) covariance multivariate gaussian
            gaussian = MultivariateNormal(loc=mu, scale_tril=torch.diag_embed(std))
            # gaussian = Independent(Normal(loc=mu, scale=std), 1)

            return gaussian
