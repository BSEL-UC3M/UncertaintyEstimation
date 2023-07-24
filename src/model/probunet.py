import torch
import torch.nn as nn
import torch.nn.init as init

from .unet import UNet
from .gaussiannet import GaussianNet
from .fcomb import FComb


class ProbabilisticUNet(nn.Module):
    """
    Probabilistic U-Net model for image segmentation with uncertainty
    estimation. It combines the deterministic segmentation provided by a U-Net
    with a probabilistic sample from the latent space encoded by a VAE to
    generate probabilistic segmentations.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Works for both RGB and grayscale images.
    n_classes : int
        Number of output classes. Refers number of labels in the segmentation
        plus the background (eg. 4 labels + background -> nclasses = 5).
    latent_dim : int
        Dimension of the latent space.
    linear_dim : int
        Input dimension of the linear layers if kohl=False. It depends on the
        input size and the parameters of the linear dimensions, and it is
        computed as H*W for 2D images and D*H*W for 3D images, being W, H and D
        the width, height and depth of the output of the last conv layer before
        the linear layer. Defauts to None.
    unet_nlayers : int
        Number of layers for both the encoder and decoder architectures.
        Defaults to 4, keeping number of layers as in the Ronneberger paper.
    vae_nlayers : int
        Number of layers for the VAE architecture. Defaults to 4, keeping
        number of layers as the regular U-Net encoder.
    unet_factor : float
        Factor to adjust the number of filters in each layer. Defaults to 1,
        with initial number of filters being 64 as in the Ronneberger paper.
    vae_factor : float
        Factor to adjust the number of filters in each layer. Defaults to 1,
        with initial number of filters being 64 as the regular U-Net.
    logits : bool
        Return a logit segmentation without applying any softmax or sigmoid layer.
    kohl : bool
        Compute the mean and the variance using Kohl et. al's original method.
        If False uses linear layers like traditional VAEs. Deor with torch.no_grad():
    model.eval()  # Set the model to evaluation modefaults to True.
    is3d : Bool
        Determines whether the network is for 2D or 3D images.

    Attributes
    ----------
    self.unet : UNet
        UNet network.
    self.prior_net : GaussianNet
        VAE network that encodes the prior latent space.
    self.posterior_net : GaussianNet
        VAE network that encodes the posterior latent space.
    self.fcomb : FComb
        Combines a sample from either a prior or posterior distribution with
        the segmentation provided by the U-Net.
    self.segmentation : torch.Tensor
        Segmentation generated by the U-Net.
    self.prior : torch.distributions.Distribution
        Prior distribution.
    self.posterior : torch.distributions.Distribution
        Posterior distribution.

    Methods
    -------
    _initialize_orthogonal_weights()
        Initializes the weights and biases of all the networks to be orthogonal.
    foward(x, y, train)
        Forward pass of the network, generating the U-Net segmentation and the
        distributions.
    sample(train)
        Sample a probabilistic segmentation.
    sample_n(n, train):
        Sample 'n' probabilistic segmentations.
    reconstruct(sample)
        Generate a probabilistic segmentation using the posterior distribution.
    """

    def __init__(self,
                 in_channels,
                 n_classes,
                 latent_dim,
                 linear_dim,
                 unet_nlayers=4,
                 vae_nlayers=4,
                 unet_factor=1,
                 vae_factor=1,
                 logits=True,
                 kohl=True,
                 is3d=False):
        super(ProbabilisticUNet, self).__init__()

        self.logits = logits
        self.n_classes = n_classes

        # Define the components of the ProbUnet
        self.unet = UNet(in_channels=in_channels,
                         n_classes=n_classes,
                         n_layers=unet_nlayers,
                         filter_factor=unet_factor,
                         dropout_p=0.2,
                         logits=logits,
                         is3d=is3d
                         )
        self.prior_net = GaussianNet(in_channels=in_channels,
                                     n_classes=n_classes,
                                     latent_dim=latent_dim,
                                     n_layers=vae_nlayers,
                                     filter_factor=vae_factor,
                                     posterior=False,
                                     kohl=kohl,
                                     linear_dim=linear_dim,
                                     is3d=is3d
                                     )
        self.posterior_net = GaussianNet(in_channels=in_channels,
                                         n_classes=n_classes,
                                         latent_dim=latent_dim,
                                         n_layers=vae_nlayers,
                                         filter_factor=vae_factor,
                                         posterior=True,
                                         kohl=kohl,
                                         linear_dim=linear_dim,
                                         is3d=is3d
                                         )
        self.fcomb = FComb(n_classes=n_classes,
                           latent_dim=latent_dim,
                           hidden_size=64 // unet_factor,
                           logits=logits,
                           is3d=is3d,
                           )

        # Initialize the weights as orthogonal
        # self._initialize_orthogonal_weights()

    def _initialize_orthogonal_weights(self):
        """
        Set the weight initialization.
        """
        # Iterate over the modules of the network
        for m in self.modules():
            # Initialize the weights of the conv layers, bn layers and linear layers
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, y, train=False):
        """
        Obtains the segmentation from the U-Net and the prior and posterior
        distributions from the autoencoders.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor representing the image.
        y : torch.Tensor
            Input tensor representing the mask.
        train : bool
            Parameter called when training, which is used to generate the
            conditional posterior. Defaults to False.

        Returns
        -------
        """
        # Get the segmentation from the U-Net
        self.segmentation = self.unet(x)

        # Find the prior distribution
        self.prior = self.prior_net(x)

        # Calculate the posterior distribution if we are training
        if train is True:
            self.posterior = self.posterior_net(x, y)

        return self.segmentation, self.prior, self.posterior

    def sample(self, train=False):
        """
        Takes a segmentation and combines it with a sample obtained from the
        prior distribution using the 'fcomb' function.

        Parameters
        ----------
        train : bool
            Parameter called when training, which is used to generate the
            sample using the reparameterization trick to allow the computation
            of gradients. Defaults to False.

        Returns
        -------
        sampled_seg : torch.Tensor
            Probabilistic segmentation.
        """

        # Sample from the prior distribution
        if train is True:
            # If train, we need to reparameterization to compute gradients
            z_prior = self.prior.rsample()
        else:
            z_prior = self.prior.sample()

        # Combine the segmentation with the sample
        sampled_seg = self.fcomb(self.segmentation, z_prior)

        return sampled_seg

    def sample_n(self, n, train=False):
        """
        Generates 'm' samples obtained from the prior distribution and combines
        then with the segmentation obtained by the U-Net using fcomb().

        Parameters
        ----------
        n : int
            Number of samples to generate.
        train : bool
            Parameter called when training, which is used to generate the
            samples using the reparameterization trick to allow the computation
            of gradients. Defaults to False.

        Returns
        -------
        sampled_segs : torch.Tensor
            Probabilistic segmentation. Has shape [m, B, L, *D, H, H]
        """

        # Initialize an empty list to store the sampled reconstructions
        sampled_reconstructions = []

        # Sample 'm' times from the prior distribution
        for _ in range(n):
            sampled_reconstruction = self.sample(train=train)
            sampled_reconstructions.append(sampled_reconstruction)

        # Stack the sampled reconstructions along a new dimension
        sampled_segs = torch.stack(sampled_reconstructions, dim=0)

        return sampled_segs

    def reconstruct(self, sample=True, z_posterior=None):
        """
        Takes a segmentation and combines it with a sample obtained from the
        posterior distribution.

        Parameters
        ----------
        sample : bool
            Chooses whether or not to sample from the prior distribution to
            obtain the sample or just use the mean of the distribution as a
            sample instead. Defaults to True (sample).
        z_posterior : torch.Tensor
            Posterior sample to be used instead of sampling again inside the
            function. Defaults to None.

        Returns
        -------
        reconstructed_seg : torch.Tensor
            Probabilistic segmentation generated from the posterior distribution.
        """

        try:
            if sample is True:
                if z_posterior is None:
                    z_posterior = self.posterior.rsample()
            else:
                z_posterior = self.posterior.loc

        except ValueError:
            # If loc returns a ValueError, sample instead
            if z_posterior is None:
                z_posterior = self.posterior.rsample()

        # Combine the segmentation with the sample
        reconstructed_seg = self.fcomb(self.segmentation, z_posterior)

        return reconstructed_seg
