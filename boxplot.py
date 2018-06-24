import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt


npzfile_interp = np.load('/Users/irinasanchez/Desktop/ds4_interp.npz')
npzfile_subpixel = np.load('/Users/irinasanchez/Desktop/ds4_subpixel.npz')
npzfile_subpixelnn = np.load('/Users/irinasanchez/Desktop/ds4_subpixelnn.npz')
npzfile_RC = np.load('/Users/irinasanchez/Desktop/ds4_RC.npz')

comb_ssim= list()
comb_ssim.append(npzfile_interp['ssim'])
comb_ssim.append(npzfile_subpixel['ssim'])
comb_ssim.append(npzfile_subpixelnn['ssim'])
comb_ssim.append(npzfile_RC['ssim'])

comb_psnr= list()
comb_psnr.append(npzfile_interp['psnr'])
comb_psnr.append(npzfile_subpixel['psnr'])
comb_psnr.append(npzfile_subpixelnn['psnr'])
comb_psnr.append(npzfile_RC['psnr'])

# t, p = ttest_rel(npzfile_subpixel['psnr'], npzfile_subpixelnn['psnr'])
# print(t)
# print(p)
# t, p = ttest_rel(npzfile_subpixel['ssim'], npzfile_subpixelnn['ssim'])
# print(t)
# print(p)
# print(npzfile_interp['psnr'].mean())
# print(npzfile_RC['psnr'].mean())
# print(npzfile_subpixel['psnr'].mean())
# print(npzfile_subpixelnn['psnr'].mean())
#
#
# fig, axes = plt.subplots(nrows=1, ncols=2)
# labels = ['CUBIC', 'SUBPIXEL', 'SUBPIXELNN', 'RC']
# # rectangular box plot
# bplot1 = axes[0].boxplot(comb_psnr,
#                          vert=True,  # vertical box alignment
#                          patch_artist=True,  # fill with color
#                          labels=labels)  # will be used to label x-ticks
# axes[0].set_title('PSNR-ds2')
#
# # notch shape box plot
# bplot2 = axes[1].boxplot(comb_ssim,
#                          vert=True,  # vertical box alignment
#                          patch_artist=True,  # fill with color
#                          labels=labels)  # will be used to label x-ticks
# axes[1].set_title('SSIM-ds2')
#
# # fill with colors
# colors = ['mediumpurple', 'lightblue', 'lightgreen', 'darksalmon']
# for bplot in (bplot1, bplot2):
#     for patch, color in zip(bplot['boxes'], colors):
#         patch.set_facecolor(color)
#
#
# plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2)
labels = ['CUBIC']
# rectangular box plot
bplot1 = axes[0].boxplot(npzfile_interp['psnr'],
                         vert=True, labels=labels)
axes[0].set_title('PSNR-ds4')

# notch shape box plot
bplot2 = axes[1].boxplot(npzfile_interp['ssim'],
                         vert=True, labels=labels)
axes[1].set_title('SSIM-ds4')

# fill with colors
# colors = ['mediumpurple', 'lightblue', 'lightgreen', 'darksalmon']
# for bplot in (bplot1, bplot2):
#     for patch, color in zip(bplot['boxes'], colors):
#         patch.set_facecolor(color)


plt.show()