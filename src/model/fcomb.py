import torch
import torch.nn as nn
import torch.nn.functional as F


class FComb(nn.Module):
    """
    Function 'fcomb' from the original paper (https://arxiv.org/abs/1806.05034).
    It takes a sample from a latent space and combines it with the output of
    the U-Net by applying 3 consecutive 1x1 convolutions followed by activation
    functions (ReLU).

    Parameters
    ----------
    n_classes : int
        Number of output classes. Refers number of labels in the segmentation
        plus the background (eg. 4 labels + background -> nclasses = 5).
    latent_dim : int
        Dimension of the latent space.
    hidden_size : int
        Intermediate size to be used in the 1x1 stack of convolutions. The
        most common value for this variable is the number of filters in the
        first encoder layer of the U-Net.
    is3d : bool
        True if input images are 3D. Defaults to False.
    logits : bool
        Return a logit segmentation without applying any softmax or sigmoid layer.

    Attributes
    ----------
    self.latent_dim : int
        Same as parameter.
    self.is3d : bool
        Same as parameter.
    self.logits : bool
        Same as parameter.
    self.conv1 = torch.nn.Module
        Convolution operation from n_classes+latent_dim to hidden_size. If is3d
        is True, the convolution is 3D, else 2D.
    self.conv2 = torch.nn.Module
        Convolution operation from hidden_size to hidden_size. If is3d is True,
        the convolution is 3D, else 2D.
    self.conv3 = torch.nn.Module
        Convolution operation from hidden_size to n_classes. If is3d is True,
        the convolution is 3D, else 2D.

    Methods
    -------
    forward(mask, z)
        Forward pass of the network.
    """

    def __init__(self, n_classes, latent_dim, hidden_size, is3d, logits=False):
        super(FComb, self).__init__()

        self.latent_dim = latent_dim
        self.is3d = is3d
        self.logits = logits

        # Define the stack of consecutive convolutional layers
        if self.is3d is True:
            self.conv1 = nn.Conv3d(n_classes + latent_dim, hidden_size, kernel_size=1)
            self.conv2 = nn.Conv3d(hidden_size, hidden_size, kernel_size=1)
            self.conv3 = nn.Conv3d(hidden_size, n_classes, kernel_size=1)

        else:
            self.conv1 = nn.Conv2d(n_classes + latent_dim, hidden_size, kernel_size=1)
            self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
            self.conv3 = nn.Conv2d(hidden_size, n_classes, kernel_size=1)

    def forward(self, mask, z):
        """
        Reshapes the latent vector to the shape of the masks and then combines
        both using a stack of 3 1x1 convolutions. It takes the dimensions of
        the mask/label and reshapes a sample 'z' from the latent space to said
        dimensions. Then, it concatenates both using the stack of convolutions.

        Parameters
        ----------
        mask : torch.Tensor
            Mask/label of the dataset.
        z : torch.Tensor
            Sample extracted from the prior/posterior distribution.

        Returns
        -------
        output : torch.Tensor
            Probabilistic mask composed of the concatenation.
        """

        # Get the dimensions of the U-Net mask
        if self.is3d is True:
            b, c, d, h, w = mask.shape
            # Turn Z into shape [b, latent_dim, 1, 1, 1]
            z = z.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            # And expand in the dimensions of the input mask to [b, latent_dim, d, h, w]
            z = z.expand(b, self.latent_dim, d, h, w)

        else:
            b, c, h, w = mask.shape
            # Turn Z into shape [b, latent_dim, 1, 1]
            z = z.unsqueeze(2).unsqueeze(3)
            # And expand in the dimensions of the input mask to [b, latent_dim, h, w]
            z = z.expand(b, self.latent_dim, h, w)

        # Concatenate mask and latent sample in the channel dimension
        x = torch.cat([mask, z], dim=1)

        # Apply the convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        output = self.conv3(x)

        if self.logits is False:
            output = torch.sigmoid(output) if x.size(1) == 2 else F.softmax(output, dim=1)

        return output
