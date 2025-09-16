import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from utils.read_write_data import save_nifti


def plot_data(imageData,labelData, plotTitle):

    fig, ax = plt.subplots(2, 2, figsize=[8, 8])
    plt.suptitle(plotTitle)
    plt.subplot(2, 2, 1)
    plt.imshow(imageData, cmap='gray', origin='lower')
    plt.subplot(2, 2, 2)
    plt.imshow(labelData, cmap='gray', origin='lower')
    plt.subplot(2, 2, 3)
    plt.imshow(imageData * labelData, cmap='gray', origin='lower')
    plt.subplot(2, 2, 4)
    plt.imshow(imageData, cmap='gray')
    plt.imshow(labelData, cmap='jet', alpha=0.5, origin='lower')

    plt.show()


def plot_vol_gt(vol, labels, thresh = 0):
    num_slices = vol.shape[2]
    for i in range(0, num_slices):
        slice_data = vol[:,:,i]
        slice_labels = labels[:,:,i]
        indices = np.nonzero(slice_labels > thresh)
        if (len(indices[0]) == 0):
            continue
        plot_data(slice_data, slice_labels, "slice: " + str(i+1))

######################################################################

data_path = '/home/bella/Phd/data/fetal_mr/brain/additional_coronal_GE/251/'
vol_filename = '251_full.nii'
gt_filename = '251_se19_seg.nii'


volume = nib.load(os.path.join(data_path, vol_filename)).get_data()
truth = nib.load(os.path.join(data_path, gt_filename)).get_data()

print("volume shape is: " + str(volume.shape))
print("labels shape is: " + str(truth.shape))

#swap x and y axes
# volume = np.swapaxes(volume, 0, 1)
# truth = np.swapaxes(truth, 0, 1)

#transpose x axes
#volume = np.flip(volume, 0)
truth = np.flip(truth, 2)
#truth = np.flip(truth, 1)

i = 15
#plot_data(volume[:,:,i],labels[:,:,i], 'slice ' + str(i))
#plot_vol_gt(volume, truth)

volume = volume[:,:, 0:truth.shape[2]]
save_nifti(np.int16(truth), os.path.join(data_path, 'truth.nii'))
save_nifti(volume, os.path.join(data_path, 'volume.nii'))