import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCoefficient(nn.Module):
    """
    Dice coefficient for either binary or multiclass segmentation tasks.
    It uses the original formulation D = 2*|X, Y| / (|X|+|Y|).

    Parameters
    ----------
    epsilon : float, optional
        Constant used to prevent division by zero and to improve stability.
        Defaults to 1e-5
    classwise : bool, optional
        Boolean parameter used to choose whether or not to return the DICE coefficient
        per class instead of the averaged coefficient, which can be useful to analyze
        per-class performance. Defaults to False.
    """

    def __init__(self, epsilon=1e-5, classwise=False):
        super(DiceCoefficient, self).__init__()
        self.epsilon = epsilon
        self.classwise = classwise

    def forward(self, input, target):
        """
        Compute the Dice coefficient.

        Parameters
        ----------
        inputs : torch.Tensor
            Predicted inputs, which can be logits or probabilities.
        targets : torch.Tensor
            Target labels.

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

        # Choose whether or not to return the loss per class. Also choose reduction.
        if self.classwise:
            dice_coef = torch.mean(dice_per_class, dim=0)
        else:
            dice_coef = torch.mean(dice_per_class)

        return dice_coef


class KLDivergence(nn.Module):
    """
    Kullback-Leibler (KL) Divergence loss module for comparing two distributions.

    Parameters
    ----------
    analytic : bool, optional
        Determines whether to use the analytic formula for calculating KL
        divergence. If True, the analytic formula using
        `torch.distributions.kl_divergence` is used. If False, the sampling
        approximation is used. Defaults to True.

    Attributes
    ----------
    analytic : bool, optional
        Same as parameter.

    Methods
    -------
    forward(prior, posterior, z_posterior=None)
        Computes the KL divergence between the prior and posterior distributions.
    """

    def __init__(self, analytic=True):
        super(KLDivergence, self).__init__()
        self.analytic = analytic

    def forward(self, prior, posterior, z_posterior=None):
        """
        Computes the KL divergence between the prior and posterior distributions.

        Parameters
        ----------
        prior : torch.distributions.Distribution
            The prior distribution.
        posterior : torch.distributions.Distribution
            The posterior distribution.
        z_posterior : torch.Tensor or None, optional
            The samples from the posterior distribution. Required when using the
            sampling approximation. Defaults to None.

        Returns
        -------
        kl : torch.Tensor
            KL divergence between the prior and posterior distributions.
        """
        if self.analytic is True:
            kl = torch.distributions.kl_divergence(posterior, prior)

        else:
            if z_posterior is None:
                z_posterior = posterior.rsample()

            log_prior = prior.log_prob(z_posterior)
            log_posterior = posterior.log_prob(z_posterior)

            kl = log_posterior - log_prior

        return kl
