import numpy as np
import nibabel as nib
from data_generation.cut_relevant_areas import find_bounding_box, cut_bounding_box
from utils.visualization import plot_vol_gt
from nilearn.image.image import _crop_img_to as crop_img_to, new_img_like
import random
from utils.image_manipulation import resize
from matplotlib import pyplot as plt
from data_generation.augment import augment_data


def crop_data(vol, mask, min_crop_size):

    vol = new_img_like(vol, vol.get_data(), affine=np.eye(4))
    mask = new_img_like(vol, mask.get_data())
    origin_size = vol.get_data().shape

    x_y_min_size = np.maximum(min_crop_size[0],min_crop_size[1])
    x_y_crop = random.randrange(x_y_min_size, origin_size[0])
    z_crop = random.randrange(min_crop_size[2], origin_size[2])

    bbox_start, bbox_end = find_bounding_box(mask.get_data())#find ground truth bounding box
    box_size = bbox_end-bbox_start
    padding_x = int(np.maximum(((x_y_crop - box_size[0])/2),0))
    padding_y = int(np.maximum(((x_y_crop - box_size[1])/2),0))
    padding_z = int(np.maximum(((z_crop - box_size[2])/2),0))

    padding = [padding_x, padding_y, padding_z]

    bbox_start = np.maximum(bbox_start - padding, 0)
    bbox_end = np.minimum(bbox_end + padding, mask.shape)

    vol = cut_bounding_box(vol, bbox_start, bbox_end)
    mask = cut_bounding_box(mask, bbox_start, bbox_end)

    vol = resize(vol,origin_size)
    mask = resize(mask, origin_size)

    return vol,mask, [bbox_start, bbox_end]

if __name__ == "__main__":
    vol = nib.load('../../../../data/brain/FR_FSE/2/volume.nii')
    mask = nib.load('../../../../data/brain/FR_FSE/2/truth.nii')
    min_crop_size = (70,70,15)
    crop_prob = 1

    ##choice implementation (50% chance to crop)
    # choice = random.randint(0, 1)
    # if(choice == 1):
    #     croped_vol, cropped_mask, [bbox_start, bbox_end] = crop_data(vol, mask, min_crop_size, crop_prob)
    #     print('cropping was selected')
    #     plot_vol_gt(croped_vol.get_data(), cropped_mask.get_data())
    # else:
    #     print('no cropping')


    croped_vol, cropped_mask, [bbox_start, bbox_end] = crop_data(vol, mask, min_crop_size)
    plot_vol_gt(croped_vol.get_data(), cropped_mask.get_data())