import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import KLDivergence


class DiceLoss(nn.Module):
    """
    Dice loss for either binary or multiclass segmentation tasks.
    It uses the original formulation D = 2*|X, Y| / (|X|+|Y|) with X being the
    prediction probabilities and Y being the ground truth/target probabilities.

    Parameters
    ----------
    reduction : str
        Type of reduction to be applied to the loss. Possible values are
        'mean', 'sum' or None (defaults to None, which means
        averaged loss).
    epsilon : float
        Constant used to prevent division by zero and to improve stability.
        Defaults to 1e-5
    classwise : bool
        Boolean parameter used to choose whether or not to return the DICE loss
        per class instead of the averaged loss, which can be useful to analyze
        per-class performance. Defaults to False.

    Attributes
    ----------
    self.reduction : str
        Same as parameters.
    self.epsilon : float
        Same as parameter.
    self.classwise : bool
        Same as parameter.

    Methods
    -------
    forward(input, target)
        Computation of the Dice loss.
    """

    def __init__(self, reduction='mean', epsilon=1e-5, classwise=False):
        super(DiceLoss, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.classwise = classwise

    def forward(self, input, target):
        """
        Compute the Dice loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Predicted inputs, which can be logits or probabilities.
        targets : torch.Tensor
            Target labels.
        reduction : str or None, optional

        Returns
        -------
        dice_loss : torch.Tensor
            Dice loss.
        """

        n_classes = input.shape[1]
        assert input.shape[1] == target.shape[1], 'Targets and inputs should match in labels'

        # Convert the inputs to probabilities if they are logits
        if not torch.all((input >= 0) & (input <= 1)):
            probs = torch.sigmoid(input) if n_classes == 2 else F.softmax(input, dim=1)
        else:
            probs = input

        # Reshape probs and targets to [batch_size, num_classes, -1]
        probs = probs.view(probs.size(0), n_classes, -1)
        targets = target.view(target.size(0), n_classes, -1)

        # Compute the intersection between samples
        intersection = torch.sum(probs * targets, dim=2)

        # Compute numerator and denominator for Dice loss
        numerator = 2 * intersection
        denominator = torch.sum(probs, dim=2) + torch.sum(targets, dim=2)

        # Compute Dice loss per class
        dice_per_class = (numerator + self.epsilon) / (denominator + self.epsilon)

        # Average Dice loss across all classes and batches
        dice_loss_per_class = 1 - dice_per_class

        # Choose whether or not to return the loss per class. Also choose reduction.
        if self.reduction == 'mean' or self.reduction is None:
            if self.classwise:
                dice_loss = torch.mean(dice_loss_per_class, dim=0)
            else:
                dice_loss = torch.mean(dice_loss_per_class)

        elif self.reduction == 'sum':
            dice_loss = torch.sum(dice_loss_per_class)

        return dice_loss


class ELBOLoss(nn.Module):
    """
    Evidence Lower Bound (ELBO) loss for Variational Autoencoders (VAEs).

    Parameters
    ----------
    beta : float
        Weight applied to the KL divergence term. Higher values of beta yield
        a more compact representation in the latent space, increasing the
        importance of the KL term.
    reduction : str or None, optional
        Specifies the reduction to apply to the loss or losses.
    analytic_kl : bool, optional
        Determines whether to use the analytic formula for calculating KL
        divergence. If True, the analytic formula is used, otherwise, the
        approximation by sampling from the posterior is used.

    Attributes
    ----------
    beta : float
        Same as parameter.
    kl : KLDivergence
        Same as parameter.
    bce : nn.BCEWithLogitsLoss
        The Binary Cross Entropy (BCE) loss module used for reconstruction loss.
    dice : DiceLoss
        The Dice loss module used for reconstruction loss.
    kl_divergence : torch.Tensor
        The calculated KL divergence term.
    reconstruction_loss : torch.Tensor
        The calculated reconstruction loss.

    Methods
    -------
    forward(reconstruction, target, prior, posterior)
        Computes the ELBO loss given the reconstruction, target, prior, and
        posterior distributions.
    """

    def __init__(self, beta, reduction=None, analytic_kl=True):
        super(ELBOLoss, self).__init__()
        self.beta = beta
        self.kl = KLDivergence(analytic=analytic_kl)
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
        self.dice = DiceLoss(reduction=reduction, classwise=False)

    def forward(self, reconstruction, target, prior, posterior):
        """
        Computes the ELBO loss given the reconstruction, target, prior, and
        posterior distributions.

        Parameters
        ----------
        reconstruction : torch.Tensor
            The reconstructed output of the VAE.
        target : torch.Tensor
            The target labels or ground truth.
        prior : torch.Tensor
            The prior distribution for the latent variables.
        posterior : torch.Tensor
            The posterior distribution for the latent variables.

        Returns
        -------
        elbo_loss : torch.Tensor
            The computed ELBO loss.
        """

        # Compute the mean batch KL divergence
        self.kl_divergence = torch.mean(self.kl(prior, posterior))

        # Compute both reconstruction losses
        bce_loss = self.bce(input=reconstruction, target=target)
        dice_loss = self.dice(input=reconstruction, target=target)

        # Define reconstruction loss
        self.reconstruction_loss = bce_loss + dice_loss

        # Define elbo loss
        elbo_loss = -(self.reconstruction_loss + self.beta * self.kl_divergence)

        return elbo_loss
