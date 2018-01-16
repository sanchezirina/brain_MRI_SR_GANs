import nibabel as nib
import os
import numpy as np
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter
from utils import aggregate
import math
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from dataset import Train_dataset

DEFAULT_SAVE_PATH_PREDICTIONS = '/work/isanchez/interpolate/ds2'

def interpolate():
    traindataset = Train_dataset(1)
    iterations = math.ceil((len(traindataset.subject_list) * 0.2))
    # 817 subjects total. De 0 a 654 training. De 654 a 817 test.
    totalpsnr = 0
    totalssim = 0
    array_psnr = np.empty(iterations)
    array_ssim = np.empty(iterations)
    batch_size = 1
    div_patches = 4
    num_patches = traindataset.num_patches
    img_width = 32  # 64
    img_height = 32  # 64
    img_depth = 23  # 46

    for i in range(0, iterations):
        XT_total = traindataset.data_true(654+i)
        XT_mask = traindataset.mask(654 + i)
        x_generator = zoom(XT_total, [1, 0.5, 0.5, 0.5])
        #x_generator = gaussian_filter(x_generator, sigma=0.5)
        volume_generated = zoom(x_generator[0], [2, 2, 2])
        volume_generated = volume_generated[:, :, :, np.newaxis]
        volume_real = XT_total[0][:, :, :, np.newaxis]
        volume_mask = aggregate(XT_mask)
        # compute metrics
        max_gen = np.amax(volume_generated)
        max_real = np.amax(volume_real)
        if max_gen > max_real:
            max = max_gen
        else:
            max = max_real
        min_gen = np.amin(volume_generated)
        min_real = np.amin(volume_real)
        if min_gen < min_real:
            min = min_gen
        else:
            min = min_real
        val_psnr = psnr(np.multiply(volume_real, volume_mask), np.multiply(volume_generated, volume_mask),
                        dynamic_range=max - min)
        array_psnr[i] = val_psnr

        totalpsnr += val_psnr
        val_ssim = ssim(np.multiply(volume_real, volume_mask), np.multiply(volume_generated, volume_mask),
                        dynamic_range=max - min, multichannel=True)
        array_ssim[i] = val_ssim
        totalssim += val_ssim
        print(val_psnr)
        print(val_ssim)
        # save volumes
        filename_gen = os.path.join(DEFAULT_SAVE_PATH_PREDICTIONS, str(i) + 'gen.nii.gz')
        img_volume_gen = nib.Nifti1Image(volume_generated, np.eye(4))
        img_volume_gen.to_filename(filename_gen)
        filename_real = os.path.join(DEFAULT_SAVE_PATH_PREDICTIONS, str(i) + 'real.nii.gz')
        img_volume_real = nib.Nifti1Image(volume_real, np.eye(4))
        img_volume_real.to_filename(filename_real)
    return array_psnr, array_ssim



if __name__ == "__main__":
    array_psnr, array_ssim = interpolate()
    print('{}{}'.format('Mean PSNR: ', array_psnr.mean()))
    print('{}{}'.format('Mean SSIM: ', array_ssim.mean()))
    print('{}{}'.format('Variance PSNR: ', array_psnr.var()))
    print('{}{}'.format('Variance SSIM: ', array_ssim.var()))
    print('{}{}'.format('Max PSNR: ', array_psnr.max()))
    print('{}{}'.format('Min PSNR: ', array_psnr.min()))
    print('{}{}'.format('Max SSIM: ', array_ssim.max()))
    print('{}{}'.format('Min SSIM: ', array_ssim.min()))
