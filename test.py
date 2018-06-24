import nibabel as nib
from scipy.ndimage.filters import gaussian_filter
from dataset import Train_dataset
from scipy.ndimage.interpolation import zoom
import os
import numpy as np


DEFAULT_SAVE_PATH_PREDICTIONS = '/work/isanchez/'

def gaus_no_prefilter():
    traindataset = Train_dataset(1)
    XT_total = traindataset.data_true(654)
    volume_real = XT_total[0][:, :, :, np.newaxis]
    filename_real = os.path.join(DEFAULT_SAVE_PATH_PREDICTIONS, 'real.nii.gz')
    img_volume_gen = nib.Nifti1Image(volume_real, np.eye(4))
    img_volume_gen.to_filename(filename_real)
    x_generator = gaussian_filter(volume_real, sigma=1)
    x_generator = zoom(x_generator, [0.5, 0.5, 0.5, 1], prefilter=False)
    filename_gen = os.path.join(DEFAULT_SAVE_PATH_PREDICTIONS, 'gaus_no_prefilter_ds2.nii.gz')
    img_volume_gen = nib.Nifti1Image(x_generator, np.eye(4))
    img_volume_gen.to_filename(filename_gen)
    x_generator = zoom(x_generator, [0.25, 0.25, 0.25, 1], prefilter=False)
    filename_gen = os.path.join(DEFAULT_SAVE_PATH_PREDICTIONS, 'gaus_no_prefilter_ds4.nii.gz')
    img_volume_gen = nib.Nifti1Image(x_generator, np.eye(4))
    img_volume_gen.to_filename(filename_gen)

def gaus_prefilter():
    traindataset = Train_dataset(1)
    XT_total = traindataset.data_true(654)
    volume_real = XT_total[0][:, :, :, np.newaxis]
    filename_real = os.path.join(DEFAULT_SAVE_PATH_PREDICTIONS, 'real.nii.gz')
    img_volume_gen = nib.Nifti1Image(volume_real, np.eye(4))
    img_volume_gen.to_filename(filename_real)
    x_generator = gaussian_filter(volume_real, sigma=1)
    x_generator = zoom(x_generator, [0.5, 0.5, 0.5, 1], prefilter=True)
    filename_gen = os.path.join(DEFAULT_SAVE_PATH_PREDICTIONS, 'gaus_prefilter_ds2.nii.gz')
    img_volume_gen = nib.Nifti1Image(x_generator, np.eye(4))
    img_volume_gen.to_filename(filename_gen)
    x_generator = zoom(x_generator, [0.25, 0.25, 0.25, 1], prefilter=True)
    filename_gen = os.path.join(DEFAULT_SAVE_PATH_PREDICTIONS, 'gaus_prefilter_ds4.nii.gz')
    img_volume_gen = nib.Nifti1Image(x_generator, np.eye(4))
    img_volume_gen.to_filename(filename_gen)


if __name__ == "__main__":
    gaus_prefilter()