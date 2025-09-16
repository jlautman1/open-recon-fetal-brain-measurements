import glob
import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import os

#from Model import config


def create_distance_map(subject_folder):
    mask_path = os.path.join(subject_folder,'truth.nii')
    distance_mask_path = os.path.join(subject_folder,'distance_mask.nii.gz')

    mask = sitk.ReadImage(mask_path)
    np_mask = sitk.GetArrayFromImage(mask)
    np_mask_invert = np_mask == 0
    np_dist = distance_transform_edt(np_mask, sampling = list(reversed(mask.GetSpacing())))
    np_dist_invert = distance_transform_edt(np_mask_invert, sampling = list(reversed(mask.GetSpacing())) )
    np_dist = [y - x for x, y in zip(np_dist, np_dist_invert)]
    np_dist = np.abs(np_dist)
    dist = sitk.GetImageFromArray(np_dist)
    dist.CopyInformation(mask)
    sitk.WriteImage(dist, distance_mask_path, True)

def run_create_distance_map(data_folder):
    for subject_folder in glob.glob(os.path.join(data_folder, "*")):
        print('prosessing folder + ' + subject_folder)
        create_distance_map(subject_folder)


if __name__ == "__main__":
    #data_folder = config["data_folder"]
    data_folder = '/home/bella/Phd/data/brain/FR_FSE_cutted/'
    run_create_distance_map(data_folder)