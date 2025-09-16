import random
import os
import pandas as pd
import nibabel as nib
import numpy as np


def get_possible_middle_slices(truth, edge_dist):
    nonzero_slices = []
    for i in range(truth.shape[2]):
        nonzero = np.nonzero(truth[:,:,i])
        if(len(nonzero[0]) > 0):
            nonzero_slices.append(i)
    min_slice = np.min(nonzero_slices)
    max_slice = np.max(nonzero_slices)

    return min_slice + edge_dist, max_slice - edge_dist


if __name__ == "__main__":
    data_path = '/home/bella/Phd/data/brain/TRUFI_axial_siemens/'
    out_file = '/home/bella/Phd/data/brain/to_annotate/variability_estimation_slices.csv'
    num_to_select = 10
    edge_dist = 4

    filenames = os.listdir(data_path)
    picked_volumes = np.random.choice(filenames, num_to_select)

    chosen_slices = set()
    for filename in picked_volumes:
        truth = nib.load(os.path.join(data_path,filename, 'truth.nii')).get_fdata()
        min_slice, max_slice = get_possible_middle_slices(truth, edge_dist)
        slice = np.random.choice(range(min_slice, max_slice))
        slice_id = filename + '_' + str(slice)
        while(slice_id in chosen_slices):
            slice = np.random.choice(range(min_slice, max_slice))
            slice_id = filename + '_' + str(slice)
        chosen_slices.add(slice_id)

    df = pd.DataFrame(chosen_slices)
    df.to_csv(out_file)