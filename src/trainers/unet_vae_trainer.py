import torch
import torch.optim as optim
import numpy as np
import time

from ..utils.misc_utils import convert_seconds
from ..utils.train_utils import l2_regularization
from ..losses import ELBOLoss
from ..metrics import DiceCoefficient


class UNetVaeTrainer:
    def __init__(self, model, beta, l2=1e-5, annealing_rate=1.2, annealing_th=0.02,
                 learning_rate=1e-4):
        self.model = model
        self.learning_rate = learning_rate
        self.l2 = l2

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.elbo = ELBOLoss(reduction='sum', beta=beta)
        self.dice = DiceCoefficient()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.dice_threshold = annealing_th
        self.annealing_rate = annealing_rate
        self.prev_train_dice = 0.0

    def kl_annealing(self, dice):
        if dice - self.prev_train_dice > self.dice_threshold:
            # Increase the KL weight for more regularization
            self.elbo.beta *= self.annealing_rate
        else:
            # Decrease the KL weight for less regularization
            self.elbo.beta /= (self.annealing_rate * 2 / 3)

        print(f'\nBeta updated to {self.elbo.beta}')

    def fit(self, trainloader, validloader, epochs):
        scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[60, 100, 120], gamma=0.1)

        train_losses = np.zeros([3, epochs])
        valid_losses = np.zeros([3, epochs])

        # Define start training time
        start_time = time.time()

        for epoch in range(epochs):
            # Set training mode
            self.model.train()

            # Define start epoch time
            epoch_time = time.time()

            # Initialize the losses
            tr_loss, tr_kl_div, tr_recon_loss = 0.0, 0.0, 0.0
            val_loss, val_kl_div, val_recon_loss = 0.0, 0.0, 0.0

            # Initialize Dice
            train_dice, valid_dice = 0., 0.

            for batch_idx, (data, target, _) in enumerate(trainloader):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                feats, mu, logvar = self.model(data, target, train=True)
                reconstruction = self.model.sample()

                # Compute the ELBO and the reconstruction and KL losses
                elbo = self.elbo(reconstruction=reconstruction, target=target,
                                 mu=mu, logvar=logvar)
                recon_loss = self.elbo.reconstruction_loss
                kl_div = self.elbo.kl_divergence

                # Compute loss
                l2_norm = (l2_regularization(self.model.prior_net) +
                           l2_regularization(self.model.fcomb)
                           )
                loss = -elbo

                # Get the total losses per batch
                tr_loss += loss.item() * data.size(0)
                tr_recon_loss += recon_loss.item() * data.size(0)
                tr_kl_div += kl_div.item() * data.size(0)

                # Compute Dice coefficient
                dice_coef = self.dice(reconstruction, target)
                train_dice += dice_coef.item()

                # Zero the gradients
                self.optimizer.zero_grad()
                # Backpropagate
                loss.backward()
                # Update model parameters
                self.optimizer.step()

            scheduler.step()

            train_losses[0, epoch] = tr_loss / len(trainloader.dataset)
            train_losses[1, epoch] = tr_recon_loss / len(trainloader.dataset)
            train_losses[2, epoch] = tr_kl_div / len(trainloader.dataset)

            train_dice = train_dice / len(trainloader.dataset)

            # Perform KL annealing
            self.kl_annealing(train_dice)
            self.prev_train_dice = train_dice

            # Set evaluation mode for validation
            self.model.eval()

            with torch.no_grad():
                for data, target, _ in validloader:
                    data, target = data.to(self.device), target.to(self.device)

                    # Forward pass
                    feats, mu, logvar = self.model(data, target, train=True)
                    sample = self.model.sample()

                    # Compute the ELBO and the reconstruction and KL losses
                    elbo = self.elbo(reconstruction=sample, target=target,
                                     mu=mu, logvar=logvar)
                    recon_loss = self.elbo.reconstruction_loss
                    kl_div = self.elbo.kl_divergence

                    loss = -elbo

                    # Compute Dice
                    dice_coef = self.dice(reconstruction, target)
                    valid_dice += dice_coef.item()

                    # Get the total losses per batch
                    val_kl_div += kl_div.item() * data.size(0)
                    val_recon_loss += recon_loss.item() * data.size(0)
                    val_loss += loss.item() * data.size(0)

                valid_losses[0, epoch] = val_loss / len(validloader.dataset)
                valid_losses[1, epoch] = val_recon_loss / len(validloader.dataset)
                valid_losses[2, epoch] = val_kl_div / len(validloader.dataset)

                valid_dice = valid_dice / len(validloader.dataset)

            # Print information after each epoch
            if (epoch % 1 == 0):
                epoch_time_str = convert_seconds(time.time() - epoch_time)

                print(f"\nEpoch {epoch + 1}/{epochs} - {epoch_time_str}\n"
                      f"-----------------------\n"
                      f"Train ELBO: {train_losses[0, epoch]:.2f}\tTrain RL: {train_losses[1, epoch]:.2f}\tTrain KL: {train_losses[2, epoch]:.2f}\tTrain Dice: {train_dice:.2f}\n"
                      f"Valid ELBO: {valid_losses[0, epoch]:.2f}\tValid RL: {valid_losses[1, epoch]:.2f}\tValid KL: {valid_losses[2, epoch]:.2f}\tValid Dice: {valid_dice:.2f}"
                      )

        # Get the time it has taken to train in HHh MMm SSs format
        train_time = convert_seconds(time.time() - start_time)
        time_per_epoch = convert_seconds((time.time() - start_time) // (epoch + 1))

        print(f"\nTotal training time:\t{train_time}\n"
              f"Average time per epoch:\t{time_per_epoch}")

        return train_losses, valid_losses

    def save(self, path):
        """
        Saves the current state of the segmentation trainer.

        Parameters
        ----------
        path : str
            The path to save the state.
        best : bool, optional
            Determines whether to save the best state (if True) or the
            current state (if False). Defaults to False.

        """
        model_state = self.model.state_dict()

        torch.save(model_state, path)
