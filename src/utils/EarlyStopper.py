import numpy as np


class EarlyStopper:
    """
    Class for early stopping during model training based on validation loss.

    Parameters
    ----------
    patience : int, optional
        # of epochs to wait for improvement in validation loss before stopping.
        The default is 1.
    min_delta : float, optional
        Minimum change in validation loss that is considered as an improvement.
        The default is 0.

    Attributes
    ----------
    patience : int
        # of epochs to wait for improvement in validation loss before stopping.
    min_delta : float
        Minimum change in validation loss that is considered as an improvement.
    counter : int
        Counter to keep track of the number of epochs without improvement.
    min_validation_loss : float
        Current minimum validation loss observed during training.

    Methods
    -------
    early_stop(validation_loss)
        Check if early stopping criterion is met based on the validation loss.
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        """
        Check if early stopping criterion is met based on the validation loss.

        Parameters
        ----------
        validation_loss : float
            The current validation loss.

        Returns
        -------
        bool
            True if the criterion is met and training should stop, else False.
        """

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False
