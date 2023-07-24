import torch.nn as nn
from .modules import DoubleConv, Down, Up, Out


class UNet(nn.Module):
    """
    U-Net model for semantic segmentation. Based on the original paper by
    Ronneberger et al. https://arxiv.org/abs/1505.04597. Works for both 2D and
    3D images. The network is composed by 4 encoder filters which encode the
    input image, one bottleneck layer to reduce dimensionality and 4 decoder
    layers to produce the segmentation. The architecture is designed such as
    the number of filters is automaticall scalable by a factor chosen by the
    user when initializing the class.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Works for both RGB and grayscale images.
    n_classes : int
        Number of output classes. Refers number of labels in the segmentation
        plus the background (eg. 4 labels + background -> nclasses = 5).
    n_layers : int
        Number of layers for both the encoder and decoder architectures.
        Defaults to 4, keeping the number of layers as in the Ronneberger paper.
    filter_factor : float
        Factor to adjust the number of filters in each layer. Defaults to 1,
        with the initial number of filters being 64 as in the Ronneberger paper.
    logits : bool
        Return a logit segmentation without applying any softmax or sigmoid layer.
    is3d : Bool
        Determines whether the network is for 2D or 3D images.

    Attributes
    ----------
    self.filters : list
        List containing the number of filters used per layer. The length of the
        list is the exact same as n_layers.
    self.is3d : bool
        Same as parameter.
    self.in_channels : int
        Same as parameter.
    self.encoder : nn.Sequential
        Encoder module responsible for downsampling the input image.
    self.bottleneck : DoubleConv
        Bottleneck module representing the central part of the U-Net.
    self.decoder : nn.Sequential
        Decoder module responsible for upsampling and recovering the spatial
        resolution.
    self.out : Out
        Output module that produces the final segmentation prediction.

    Methods
    -------
    _encoder(in_channels, filters)
        Function used to create the encoder path of the U-Net
    _decoder(filters)
        Function used to create the decoder path of the U-Net
    forward(x)
        Forward pass of the U-Net network.
    """

    def __init__(self, in_channels, n_classes, n_layers=4, filter_factor=1,
                 dropout_p=0, logits=False, is3d=False):
        super(UNet, self).__init__()

        base_filters = 64 // filter_factor
        self.filters = [base_filters * (2 ** i) for i in range(n_layers)]
        self.is3d = is3d
        self.in_channels = in_channels
        self.dropout_p = dropout_p

        self.encoder = self._encoder()
        self.bottleneck = DoubleConv(self.filters[-1], self.filters[-1] * 2,
                                     dropout_p=dropout_p, is3d=is3d)
        self.decoder = self._decoder()
        self.out = Out(self.filters[0], n_classes, logits=logits, is3d=is3d)

    def _encoder(self):
        """
        Function used to create the encoder path of the U-Net. It takes a list
        containing the number of features, iterates through it and appends each
        encoder layer to an empty list. Then, the list is used to create a
        torch.nn.Sequential object that can be used in the forward pass.

        Returns
        -------
        encoder : nn.Sequential
            Encoder path of the U-Net.
        """

        # Create empty list to store each encoder layer
        encoder_layers = []

        # First decoder layer takes the input channels, so create it separately
        encoder_layers.append(DoubleConv(self.in_channels, self.filters[0],
                                         dropout_p=self.dropout_p, is3d=self.is3d))

        # Create the rest of the layers using a for loop
        for i in range(len(self.filters) - 1):
            encoder_layers.append(Down(self.filters[i], self.filters[i + 1],
                                       dropout_p=self.dropout_p, is3d=self.is3d))

        encoder = nn.Sequential(*encoder_layers)

        return encoder

    def _decoder(self):
        """
        Function uses to create the decoder path of the U-Net. It takes a list
        containing the number of features, iterates through it and appends each
        decoder layer to an empty list. Then, the list is used to create a
        torch.nn.Sequential object that can be used in the forward pass.

        Returns
        -------
        decoder : nn.Sequential
            Decoder path of the U-Net.
        """

        # Create empty list to store each decoder layer
        decoder_layers = []

        # Create the layers using a for loop
        for i in range(len(self.filters) - 1, -1, -1):
            decoder_layers.append(Up(self.filters[i] * 2, self.filters[i],
                                     dropout_p=self.dropout_p, is3d=self.is3d))

        decoder = nn.Sequential(*decoder_layers)

        return decoder

    def forward(self, x):
        """
        Forward pass of the U-Net model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        out : torch.Tensor
            Output tensor representing the segmentation prediction. It is
            a probabilistic segmentation since either a Sigmoid or Softmax
            function is applied.
        """

        # Create an empty list to store the encoders for the skip-connections
        encoders = []
        for encoder in self.encoder:
            # Apply encoding layer
            x = encoder(x)
            encoders.append(x)

        # Bottleneck layer
        x = self.bottleneck(x)

        for i, decoder in enumerate(self.decoder):
            # Apply decoding layer together with skip-connections
            x = decoder(x, encoders[-(i + 1)])

        # Output layer
        out = self.out(x)

        return out
