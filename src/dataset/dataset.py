import os
from torch.utils.data import Dataset
import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


class UryToxDataset(Dataset):
    """
    Dataset class for UryTox dataset.

    Parameters
    ----------
    root_dir : str
        Root directory of the dataset.
    organ : str
        Choose which organ to include in the labels. Options are 'r' (rectum),
        'b' (bladder), 'p' (prostate), 'v' (seminal vesicles) and 'a' (all).
        Letters can be concatenated to select multiple organs (e.g. 'rb' will
        include both rectum and bladder'). Defaults to 'a' (all)
    input_dim : tuple, optional
        Tuple specifying the input dimensions of the images. Defaults to
        (224, 224, 224).
    nlabels : int, optional
        Number of labels. Defaults to 4.
    transform : callable, optional
        Optional data augmentation transformations. Defaults to None.

    Attributes
    ----------
    self.root_dir : str
        Same as parameter.
    self.organ : str
        Same as parameter.
    self.input_dim : int
        Same as parameter.
    self.nlabels : int, optional
        Same as parameter.
    self.transform : callable, optional
        Same as parameter.
    self.images_path : str
        Path where the images are located.
    self.labels_path : str
        Path where the labels are located.
    self.images : list
        List containing the paths for each and every image.
    self.labels : list
        List containing the paths for each and every label.

    Methods
    -------
    __len__()
        Get dataset length.
    __getitem(idx)__
        Read and retrieve a single pair of label and mask.
    plotitem(idx, slc)
        Plot a the selected slice of single image and pair mask in the 3 planes.
    """

    def __init__(self, root_dir, organ='a', input_dim=(224, 224, 224), nlabels=4, transform=None):
        super(UryToxDataset, self).__init__()

        self.root_dir = root_dir
        self.organ = organ
        self.input_dim = input_dim
        self.nlabels = nlabels
        self.transform = transform

        # Define the path for images and their labels
        self.images_path = self.root_dir + '/images/'
        self.labels_path = self.root_dir + '/labels/'

        # Sort the list of images and labels so that they match
        self.images = sorted(os.listdir(self.images_path))
        self.labels = sorted(os.listdir(self.labels_path))

        # Ensure images and labels have the same dimension
        assert len(self.images) == len(self.labels), "Number of images and labels don't match"

    def __len__(self):
        """
        Get the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """

        return len(self.images)

    def __getitem__(self, idx):
        """
        Get an item from the dataset at the given index.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        Returns
        -------
        image_tensor : torch.tensor
            Image tensor of shape (C, D, H, W).
        label_tensor : torch.tensor
            Label tensor of shape (C, D, H, W).
        img_name : str
            Filename of the image retrieved.
        """

        # Select the item of interest
        img_name = self.images[idx]
        label_name = self.labels[idx]

        # Read the images and labels and convert them to numpy arrays
        image = sitk.ReadImage(self.images_path + img_name)
        label = sitk.ReadImage(self.labels_path + label_name)

        # Transform them if specified
        if self.transform:
            image, label = self.transform((image, label))

        # Flip the images so that they have the correct view
        image = sitk.Flip(image, [False, False, True])
        label = sitk.Flip(label, [False, False, True])

        image = sitk.GetArrayFromImage(image)
        label = sitk.GetArrayFromImage(label)

        # Normalize the image
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # Convert label to one-hot encoding
        label = label.astype(int)
        label[label >= self.nlabels] = 0
        label_one_hot = torch.nn.functional.one_hot(torch.from_numpy(label),
                                                    self.nlabels + 1).float()

        # Change the order so that it is CxDxHxW instead of DxHxWxC
        label_tensor = label_one_hot.permute(3, 0, 1, 2)

        # Select individual organs if desired
        if 'a' not in self.organ:
            # 0 background, 1 rectum, 2 bladder, 3 prostate, 4 seminal vesicles
            organs = dict.fromkeys(['0', 'r', 'b', 'p', 'v'])
            # Split the label into individual organs
            labels = np.split(label_tensor, 5, axis=0)
            # Fill the empty dictionary
            organs_dict = {org: label for org, label in zip(organs, labels)}
            # Select the desired organ/s plus the background
            selected_tensors = [organs_dict[org] for org in '0' + self.organ]
            # Create the label tensor
            label_tensor = torch.cat(selected_tensors, dim=0)

        # Convert image to PyTorch tensor so that dimensions are CxWxHxD
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)

        return image_tensor, label_tensor, img_name

    def plotitem(self, idx, slc=None):
        """
        Function to show the 3 anatomical planes of a selected image together
        with its ground truth.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.
        slc : int
            Slice of the item to show. Defaults to None (middle slice)
        """

        image, label, name = self.__getitem__(idx)

        # Undo one-hot encoding to plot all labels at once
        label = torch.argmax(label, dim=0, keepdim=True)

        # Select the middle slice if not specified
        if slc is None:
            slc = image.shape[2] // 2

        # Plot the images and labels
        fig, axs = plt.subplots(2, 3, figsize=(10, 8))

        axs[0, 0].imshow(image[0, :, :, slc], cmap=plt.cm.Greys_r)
        axs[0, 0].set_title(f'Sagittal plane image {name.strip(".nii.gz")}')

        axs[0, 1].imshow(image[0, :, slc, :], cmap=plt.cm.Greys_r)
        axs[0, 1].set_title(f'Coronal Plane {name.strip(".nii.gz")}')

        axs[0, 2].imshow(image[0, slc, :, :], cmap=plt.cm.Greys_r)
        axs[0, 2].set_title(f'Axial plane {name.strip(".nii.gz")}')

        axs[1, 0].imshow(label[0, :, :, slc], cmap='viridis')
        axs[1, 0].set_title(f'Sagittal plane label {name.strip("_0000.nii.gz")}')

        axs[1, 1].imshow(label[0, :, slc, :], cmap='viridis')
        axs[1, 1].set_title(f'Coronal Plane {name.strip("_0000.nii.gz")}')

        axs[1, 2].imshow(label[0, slc, :, :], cmap='viridis')
        axs[1, 2].set_title(f'Axial plane {name.strip("_0000.nii.gz")}')

        plt.tight_layout()
        plt.show()
