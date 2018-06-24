import numpy as np
import nibabel as nib
import math
import os
from skimage.util import view_as_windows
import csv


class Train_dataset(object):
    def __init__(self, batch_size, overlapping=1):
        self.batch_size = batch_size
        self.data_path = '/imatge/isanchez/projects/neuro/ADNI-Screening-1.5T'
        self.subject_list = os.listdir(self.data_path)
        self.subject_list = np.delete(self.subject_list, 120)
        self.heigth_patch = 112  # 128
        self.width_patch = 112  # 128
        self.depth_patch = 76  # 92
        self.margin = 16
        self.overlapping = overlapping
        self.num_patches = (math.ceil((224 / (self.heigth_patch)) / (self.overlapping))) * (
            math.ceil((224 / (self.width_patch)) / (self.overlapping))) * (
                               math.ceil((152 / (self.depth_patch)) / (self.overlapping)))

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
            if subject != 'ADNI_SCREENING_CLINICAL_FILE_08_02_17.csv':
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
    #subject_list = os.listdir(data_path)
    #index = subject_list.index('ADNI_SCREENING_CLINICAL_FILE_08_02_17.csv')
    #print(subject_list[index])
    cont_AD = 0
    cont_MCI = 0
    cont_NC = 0
    subject_list = os.listdir(data_path)
    print(subject_list)
    with open(os.path.join(data_path, 'ADNI_SCREENING_CLINICAL_FILE_08_02_17.csv')) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            print(row)
            type = row[4]
            print(type)
            if type == 'AD':
                cont_AD += 1
                print('contAD')
            elif type == 'MCI':
                cont_MCI += 1
                print('contMCI')
            else:
                cont_NC += 1
                print('contNC')
    print('Number of AD: %2d' % cont_AD)
    print('Number of MCI: %2d' % cont_MCI)
    print('Number of NC: %2d' % cont_NC)
    print(cont_AD + cont_MCI + cont_NC)
