import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleConv(nn.Module):
    """
    SingleConv module performs a single convolution operation followed by batch
    normalization (or not) and activation function.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    k_size : int
        Convolution kernel size. Defaults to 3.
    stride : int
        Convolution stride. Defaults to 1.
    pad : int
        Convolution padding. Defaults to 1.
    dropout_prob : float
        Dropout probability. Defaults to 0.5.
    activation : str
        Activation function to use. Options: 'l' (LeakyReLU), 'g' (GELU),
        'e' (ELU), 'r' (ReLU). Defaults to 'l'.
    bn : bool
        Specifies whether to use batch normalization. Defaults to True.
    is3d : bool
        Specifies whether to use 3D convolution or 2D convolution. Defaults to
        False.

    Attributes
    ----------
    self.activation : str
        Same as parameter.
    self.is3d : bool
        Same as parameter.
    self.bn : bool
        Same as parameter.
    self.conv = torch.nn.Module
        Convolution operation. If is3d is True, the convolution is 3D, else 2D.
    self.batch_norm = torch.nn.Module
        Batch normalization operation. If is3d is True is 3D, else 2D.
    self.dropout : torch.nn.Module
        Dropout layer.

    Methods
    -------
    forward(x)
        Forward pass
    """

    def __init__(self, in_channels, out_channels, k_size=3, stride=1, pad=1,
                 activation='l', dropout_p=0.2, bn=True, is3d=False):

        super(SingleConv, self).__init__()

        self.activation = activation
        self.is3d = is3d
        self.bn = bn

        if is3d is True:
            self.conv = nn.Conv3d(in_channels, out_channels,
                                  kernel_size=k_size, stride=stride, padding=pad)
            self.batch_norm = nn.BatchNorm3d(num_features=out_channels)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=k_size, stride=stride, padding=pad)
            self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        """
        Forward pass of the SingleConv module (convolution, *batch normalization
        and activation function).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        x : torch.Tensor
            Output tensor.
        """

        x = self.conv(x)

        if self.bn:
            x = self.batch_norm(x)

        if self.activation == 'l':
            x = F.leaky_relu(x)
        elif self.activation == 'g':
            x = F.gelu(x)
        elif self.activation == 'e':
            x = F.elu(x)
        else:
            x = F.relu(x)

        x = self.dropout(x)

        return x


class DoubleConv(nn.Module):
    """
    DoubleConv module performs two consecutive SingleConvs (convolution,
    *batch norm., activation, convolution, *batch norm., activation).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    k_size : int
        Convolution kernel size. Defaults to 3.
    stride : int
        Convolution stride. Defaults to 1.
    pad : int
        Convolution padding. Defaults to 1.
    bn : bool
        Specifies whether to use batch normalization. Defaults to True.
    is3d : bool
        Specifies whether to use 3D convolution or 2D convolution. Defaults to
        False.

    Attributes
    ----------
    self.conv1 : SingleConv
        Convolution followed by batch normalization and activation function.
    self.conv2 : SingleConv
        Convolution followed by batch normalization and activation function.

    Methods
    -------
    forward(x)
        Forward pass
    """

    def __init__(self, in_channels, out_channels, k_size=3, stride=1, pad=1,
                 dropout_p=0.2, bn=True, is3d=False):

        super(DoubleConv, self).__init__()

        self.conv1 = SingleConv(in_channels, out_channels,
                                k_size=k_size, stride=stride, pad=pad,
                                dropout_p=dropout_p, bn=bn, is3d=is3d)
        self.conv2 = SingleConv(out_channels, out_channels,
                                k_size=k_size, stride=stride, pad=pad,
                                dropout_p=dropout_p, bn=bn, is3d=is3d)

    def forward(self, x):
        """
        Forward pass of the DoubleConv module (convolution, *batch norm.,
        activation, convolution, *batch norm., activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        x : torch.Tensor
            Output tensor.
        """

        x = self.conv1(x)
        x = self.conv2(x)

        return x


class Down(nn.Module):
    """
    Down module performs downsampling through a double convolution followed by
    max pooling.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn : bool
        Specifies whether to use batch normalization. Defaults to True.
    is3d : bool
        Specifies whether to use 3D convolution or 2D convolution. Defaults to
        False.

    Attributes
    ----------
    self.double_conv : DoubleConv
        DoubleConv operation
    self.pool : torch.nn.Module
        Max pooling operation (3D if is3d=True, else 2D).

    Methods
    -------
    forward(x)
        Forward pass
    """

    def __init__(self, in_channels, out_channels, dropout_p=0.2, bn=True, is3d=False):
        super(Down, self).__init__()

        self.double_conv = DoubleConv(in_channels, out_channels,
                                      dropout_p=dropout_p, bn=bn, is3d=is3d)

        if is3d is True:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward pass of the Down module (double convolution and max pooling).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        x : torch.Tensor
            Output tensor.
        """

        x = self.double_conv(x)
        x = self.pool(x)

        return x


class Up(nn.Module):
    """
    Up module performs upsampling through transposed convolution followed by
    concatenation and double convolution.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn : bool
        Specifies whether to use batch normalization. Defaults to True.
    is3d : bool
        Specifies whether to use 3D convolution or 2D convolution. Defaults to
        False.

    Attributes
    ----------
    self.is3d : bool
        Same as parameter
    self.up : torch.nn.Module
        Transposed convolution. 3D if is3d=True else 2D.
    self.double_conv : DoubleConv
        DoubleConv operation

    Methods
    -------
    forward(x)
        Forward pass
    """

    def __init__(self, in_channels, out_channels, dropout_p=0.2, bn=True, is3d=False):
        super(Up, self).__init__()

        self.is3d = is3d

        if is3d is True:
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.double_conv = DoubleConv(in_channels, out_channels, dropout_p=dropout_p,
                                      bn=bn, is3d=is3d)

    def forward(self, x1, x2):
        """
        Forward pass of the Up module (transposed convolution and max pooling)
        with skip-connections between decoder and encoder vectors.

        Parameters
        ----------
        x1 : torch.Tensor
            Upscaled input tensor.
        x2 : torch.Tensor
            Skip-connection input tensor.

        Returns
        -------
        x : torch.Tensor
            Output tensor.
        """

        x1 = self.up(x1)

        # Perform padding
        if self.is3d:
            # Compute the differences between tensors
            diffZ = x2.shape[2] - x1.shape[2]
            diffY = x2.shape[3] - x1.shape[3]
            diffX = x2.shape[4] - x1.shape[4]

            # Pad x1 with the differences
            x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2,
                            diffZ // 2, diffZ - diffZ // 2))

        else:
            # Compute the differences between tensors
            diffY = x2.shape[2] - x1.shape[2]
            diffX = x2.shape[3] - x1.shape[3]

            # Pad x1 with the differences
            x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2))

        # Concatenate the two vectors
        x = torch.cat([x2, x1], dim=1)

        return self.double_conv(x)


class Out(nn.Module):
    """
    Out module performs a final convolution operation followed by activation
    function which is either softmax if multi-class or sigmoid if the problem
    involves only two classes (background and target).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    logits : bool
        Determine if the output should be a logit mask instead of each pixel
        representing a probability. Defaults to False.
    is3d : bool
        Specifies whether to use 3D convolution or 2D convolution. Defaults to
        False.

    Attributes
    ----------
    self.logits : bool
        Same as parameter.
    self.conv = torch.nn.Module
        Convolution operation. If is3d is True, the convolution is 3D, else 2D.

    Methods
    -------
    forward(x)
        Forward pass
    """

    def __init__(self, in_channels, out_channels, logits=False, is3d=False):
        super(Out, self).__init__()

        self.logits = logits

        if is3d is True:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the Out module (convolution and activation).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        x : torch.Tensor
            Output tensor.
        """

        x = self.conv(x)

        if self.logits is False:
            x = torch.sigmoid(x) if x.size(1) == 2 else F.softmax(x, dim=1)

        return x
