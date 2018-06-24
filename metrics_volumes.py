import nibabel as nib
import os
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

array_psnr = []
array_ssim = []
path_volumes = '/work/isanchez/predictions/volumesTF/ds4-gdl-lrdecay-ok/RC'
#path_volumes = '/work/isanchez/interpolate/ds4'
totalpsnr = 0
totalssim = 0

for i in range(0, 163):
    filename_gen = os.path.join(path_volumes, str(i) + 'gen.nii.gz')
    volume_generated = nib.load(filename_gen)
    data_volume_gen = np.array(volume_generated.get_data(), dtype=np.float64)
    filename_real = os.path.join(path_volumes, str(i) + 'real.nii.gz')
    volume_real = nib.load(filename_real)
    data_volume_real = np.array(volume_real.get_data(), dtype=np.float64)
    max_gen = np.amax(data_volume_gen)
    max_real = np.amax(data_volume_real)
    if max_gen > max_real:
        max = max_gen
    else:
        max = max_real
    min_gen = np.amin(data_volume_gen)
    min_real = np.amin(data_volume_real)
    if min_gen < min_real:
        min = min_gen
    else:
        min = min_real
    val_psnr = psnr(data_volume_gen, data_volume_real, dynamic_range=max - min)
    array_psnr.append(val_psnr)
    val_ssim = ssim(data_volume_gen, data_volume_real, dynamic_range=max - min, multichannel=True)
    array_ssim.append(val_ssim)
    print(val_psnr)
    print(val_ssim)

print('{}{}'.format('PSNR: ', array_psnr))
print('{}{}'.format('SSIM: ', array_ssim))
print('{}{}'.format('Mean PSNR: ', np.mean(array_psnr)))
print('{}{}'.format('Mean SSIM: ', np.mean(array_ssim)))
print('{}{}'.format('Variance PSNR: ', np.var(array_psnr)))
print('{}{}'.format('Variance SSIM: ', np.var(array_ssim)))
print('{}{}'.format('Max PSNR: ', np.max(array_psnr)))
print('{}{}'.format('Min PSNR: ', np.min(array_psnr)))
print('{}{}'.format('Max SSIM: ', np.max(array_ssim)))
print('{}{}'.format('Min SSIM: ', np.min(array_ssim)))
print('{}{}'.format('Median PSNR: ', np.median(array_psnr)))
print('{}{}'.format('Median SSIM: ', np.median(array_ssim)))