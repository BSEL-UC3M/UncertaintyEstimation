import SimpleITK as sitk
import numpy as np


class Resize(object):
    """
    Resize the image and label data using a scale factor.

    Parameters
    ----------
    scale_factor : float
        Changes the images and labels sizes to scale_factor * (W, H, D)

    Returns
    -------
    image : SimpleITK.Image
        Resized image
    label : SimpleITK.Image
        Resized label
    """

    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, sample):
        image, label = sample

        # Get original size and spacing of the image
        original_size = np.array(image.GetSize())
        original_spacing = np.array(image.GetSpacing())

        # Calculate new size and spacing based on scale factor for the image
        new_size = np.round(original_size * self.scale_factor).astype(int)
        new_spacing = original_spacing / self.scale_factor

        # Resample the image to the desired size and spacing
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing.tolist())
        resampler.SetSize(new_size.tolist())
        resampled_image = resampler.Execute(image)

        # Resample the label separately to the desired size and spacing
        resampler_label = sitk.ResampleImageFilter()
        resampler_label.SetOutputSpacing(new_spacing.tolist())
        resampler_label.SetSize(new_size.tolist())
        resampled_label = resampler_label.Execute(label)

        return resampled_image, resampled_label
