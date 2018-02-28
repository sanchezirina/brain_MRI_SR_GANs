import numpy as np
import nibabel as nib
import math
import os
from skimage.util import view_as_windows
import csv
from scipy.ndimage.filters import gaussian_filter


class Train_dataset(object):
    def __init__(self, batch_size, overlapping=1):

        self.batch_size = batch_size
        self.data_path = '/imatge/isanchez/projects/neuro/ADNI-Screening-1.5T'
        self.subject_list = []
        self.heigth_patch = 112  # 128
        self.width_patch = 112  # 128
        self.depth_patch = 76  # 92
        self.margin = 16
        self.overlapping = overlapping
        self.num_patches = (math.ceil((224 / (self.heigth_patch)) / (self.overlapping))) * (
            math.ceil((224 / (self.width_patch)) / (self.overlapping))) * (
                               math.ceil((152 / (self.depth_patch)) / (self.overlapping)))
        with open(os.path.join(data_path, 'ADNI_SCREENING_CLINICAL_FILE_08_02_17.csv')) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                if row[4] != 'AD':
                    self.subject_list.append(row[0])
        self.subject_list = np.delete(self.subject_list, 0)

    def mask(self, iteration):
        subject_batch = self.subject_list[iteration * self.batch_size:self.batch_size + (iteration * self.batch_size)]
        subjects_true = np.empty([self.batch_size, 256, 256, 184])
        i = 0
        for subject in subject_batch:
            if subject != 'ADNI_SCREENING_CLINICAL_FILE_08_02_17.csv':
                filename = os.path.join(self.data_path, subject)
                filename = os.path.join(filename, 'T1_brain_extractedBrainExtractionMask.nii.gz')
                proxy = nib.load(filename)
                data = np.array(proxy.dataobj)

                # change for math.ceil

                paddwidthr = int((256 - proxy.shape[0]) / 2)
                paddheightr = int((256 - proxy.shape[1]) / 2)
                paddepthr = int((184 - proxy.shape[2]) / 2)

                if (paddwidthr * 2 + proxy.shape[0]) != 256:
                    paddwidthl = paddwidthr + 1
                else:
                    paddwidthl = paddwidthr

                if (paddheightr * 2 + proxy.shape[1]) != 256:
                    paddheightl = paddheightr + 1
                else:
                    paddheightl = paddheightr

                if (paddepthr * 2 + proxy.shape[2]) != 184:
                    paddepthl = paddepthr + 1
                else:
                    paddepthl = paddepthr

                data_padded = np.pad(data,
                                     [(paddwidthl, paddwidthr), (paddheightl, paddheightr), (paddepthl, paddepthr)],
                                     'constant', constant_values=0)
                subjects_true[i] = data_padded
                i = i + 1
        mask = np.empty(
            [self.batch_size * self.num_patches, self.width_patch + self.margin, self.heigth_patch + self.margin,
             self.depth_patch + self.margin, 1])
        i = 0
        for subject in subjects_true:
            patch = view_as_windows(subject, window_shape=(
                (self.width_patch + self.margin), (self.heigth_patch + self.margin), (self.depth_patch + self.margin)),
                                    step=(self.width_patch - self.margin, self.heigth_patch - self.margin,
                                          self.depth_patch - self.margin))
            for d in range(patch.shape[0]):
                for v in range(patch.shape[1]):
                    for h in range(patch.shape[2]):
                        p = patch[d, v, h, :]
                        p = p[:, np.newaxis]
                        p = p.transpose((0, 2, 3, 1))
                        mask[i] = p
                        i = i + 1
        return mask

    def patches_true(self, iteration):
        subjects_true = self.data_true(iteration)
        patches_true = np.empty(
            [self.batch_size * self.num_patches, self.width_patch + self.margin, self.heigth_patch + self.margin,
             self.depth_patch + self.margin, 1])
        i = 0
        for subject in subjects_true:
            patch = view_as_windows(subject, window_shape=(
                (self.width_patch + self.margin), (self.heigth_patch + self.margin), (self.depth_patch + self.margin)),
                                    step=(self.width_patch - self.margin, self.heigth_patch - self.margin,
                                          self.depth_patch - self.margin))
            for d in range(patch.shape[0]):
                for v in range(patch.shape[1]):
                    for h in range(patch.shape[2]):
                        p = patch[d, v, h, :]
                        p = p[:, np.newaxis]
                        p = p.transpose((0, 2, 3, 1))
                        # 128x128x92
                        patches_true[i] = p
                        i = i + 1
        return patches_true

    def data_true(self, iteration):
        subject_batch = self.subject_list[iteration * self.batch_size:self.batch_size + (iteration * self.batch_size)]
        subjects = np.empty([self.batch_size, 224, 224, 152])
        i = 0
        for subject in subject_batch:
                filename = os.path.join(self.data_path, subject)
                filename = os.path.join(filename, 'T1_brain_extractedBrainExtractionBrain.nii.gz')
                proxy = nib.load(filename)
                data = np.array(proxy.dataobj)

                # change for math.ceil

                paddwidthr = int((256 - proxy.shape[0]) / 2)
                paddheightr = int((256 - proxy.shape[1]) / 2)
                paddepthr = int((184 - proxy.shape[2]) / 2)

                if (paddwidthr * 2 + proxy.shape[0]) != 256:
                    paddwidthl = paddwidthr + 1
                else:
                    paddwidthl = paddwidthr

                if (paddheightr * 2 + proxy.shape[1]) != 256:
                    paddheightl = paddheightr + 1
                else:
                    paddheightl = paddheightr

                if (paddepthr * 2 + proxy.shape[2]) != 184:
                    paddepthl = paddepthr + 1
                else:
                    paddepthl = paddepthr

                data_padded = np.pad(data,
                                     [(paddwidthl, paddwidthr), (paddheightl, paddheightr), (paddepthl, paddepthr)],
                                     'constant', constant_values=0)

                subjects[i] = data_padded[16:240, 16:240, 16:168]  # remove background
                i = i + 1
        return subjects


if __name__ == '__main__':
    data_path = '/imatge/isanchez/projects/neuro/ADNI-Screening-1.5T'
    DEFAULT_SAVE_PATH_VOLUMES = '/work/isanchez/predictions/'
    dataset = Train_dataset(batch_size=1)
    print(dataset.subject_list)
    print(len(dataset.subject_list))
    patches = dataset.patches_true(iteration=0)
    for index, patch in enumerate(patches):
        filename_true = os.path.join(DEFAULT_SAVE_PATH_VOLUMES, str(index) + 'true.nii.gz')
        img_patch_true = nib.Nifti1Image(patch, np.eye(4))
        img_patch_true.to_filename(filename_true)
        filename_gaus = os.path.join(DEFAULT_SAVE_PATH_VOLUMES, str(index) + 'gaus.nii.gz')
        img_patch_gaus = nib.Nifti1Image(gaussian_filter(patch, sigma=1), np.eye(4))
        img_patch_gaus.to_filename(filename_gaus)