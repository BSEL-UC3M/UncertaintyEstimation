import os
import torch
from torch.utils.data import DataLoader, random_split
from src.dataset.dataset import UryToxDataset
from src.dataset.transforms import Resize

# ProbUnet training
from src.model.probunet import ProbabilisticUNet
from src.trainers.probunet_trainer import ProbUnetTrainer

# U-Net training
from src.trainers.segtrainer import SegmentationTrainer
from src.model.unet import UNet
from src.losses import DiceLoss


def load_urytox(path):
    """
    Function used to load the UryTox dataset.

    Parameters
    ----------
    path : str
        Main path where the images are located.

    Returns
    -------
    dataset : UryToxDataset
        Dataset class for the UryTox images.
    """

    # Choose the path for the data
    path = path + '/data/UryToxFiltered'

    # Load the data
    dataset = UryToxDataset(path, organ='b', transform=Resize(scale_factor=0.5))

    dataset.plotitem(4, 56)

    # Check that the sizes are correct
    image, label, _ = dataset[0]

    print(f'There are {dataset.__len__()} pairs of images and labels')
    print(f'The images have shape {image.shape}\nThe labels have shape {label.shape}\n')

    return dataset


def create_dataloaders(dataset, seed, batch_size, train_size, val_size=0.15):
    """
    Function used to create the dataloaders for training.

    Parameters
    ----------
    dataset :
        Dataset from which the dataloaders are created.
    seed : torch.Generator().manual_seed
        PyTorch seed to ensure reproducible results.
    batch_size : int
        Batch size.
    train_size : float
        Proportion of the whole data that will be used as training. Has to be
        between 0 and 1.
    val_size : float
        Proportion of the whole data that will be used for validation during
        training. Has to be between 0 and 1-train_size. Defaults to 0.15.

    Returns
    -------
    trainloader : torch.DataLoader
        Training data.
    validloader : torch.DataLoader
        Validation data.
    testloader : torch.DataLoader
        Testing data.
    """

    # Defining the size of each subset
    train_size = int(len(dataset) * train_size)
    valid_size = int(len(dataset) * val_size)
    test_size = len(dataset) - train_size - valid_size

    # Splitting the dataset into training, validation, and testing sets
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, valid_size, test_size], generator=seed)

    # Creating data loaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(val_dataset, batch_size=batch_size)
    testloader = DataLoader(test_dataset, batch_size=batch_size)

    # Check that data is loaded correctly
    # for batch_idx, (data, target, name) in enumerate(trainloader):
    #     print(
    #         f'Batch {batch_idx + 1} - data size: {data.size()}, target size: {target.size()}')

    return trainloader, validloader, testloader


def train_unet(trainloader, validloader, testloader):
    loss = DiceLoss()
    model = UNet(in_channels=1, n_classes=2, filter_factor=2, logits=True, is3d=True)
    trainer = SegmentationTrainer(model, classes=2, loss=loss, learning_rate=1e-4)

    tr_loss, val_loss = trainer.fit(trainloader, validloader, epochs=100)
    trainer.save(os.path.join(os.getcwd(), 'results/trained_models/test_bladder_unet.pth'))


def train_probunet(trainloader, validloader, testloader):
    model = ProbabilisticUNet(in_channels=1,
                              n_classes=2,
                              latent_dim=6,
                              linear_dim=14 ** 3,
                              unet_factor=2,
                              vae_factor=2,
                              logits=True,
                              is3d=True)

    trainer = ProbUnetTrainer(model, beta=0, learning_rate=1e-3)

    tr_loss, val_loss = trainer.fit(trainloader, validloader, epochs=100)

    trainer.save(os.path.join(os.getcwd(), 'results/trained_models/test_bladder_justunet.pth'))


def main():
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'The device being used is {device}\n')

    # Set the seeds
    seed = torch.Generator().manual_seed(42)
    # torch.manual_seed(42)

    # Load the dataset
    urytox_dataset = load_urytox(os.getcwd())

    # Create the dataloaders
    trainloader, validloader, testloader = create_dataloaders(
        urytox_dataset, seed, batch_size=2, train_size=0.7)

    train_unet(trainloader, validloader, testloader)


if __name__ == '__main__':
    main()
