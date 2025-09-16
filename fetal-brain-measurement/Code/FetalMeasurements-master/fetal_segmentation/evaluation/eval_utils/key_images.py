from collections import namedtuple
import nibabel as nib
import os
from utils.visualization import get_plot_gt_res_overlays, prepare_for_plotting
import numpy as np
from PIL import Image



def is_local_minimum(vol_slice_eval, index):
    prev_index = index - 1
    next_index = index + 1
    curr_val = vol_slice_eval[index]

    if((prev_index not in vol_slice_eval) and (next_index in vol_slice_eval)):#if it is the first slice, check that next slice has higher dice
        if(vol_slice_eval[next_index] > curr_val):
            return True
        else:
            return False

    if((next_index not in vol_slice_eval) and (prev_index in vol_slice_eval)):#if it is the last slice, check that previous slice has higher dice
        if(vol_slice_eval[prev_index] > curr_val):
            return True
        else:
            return False

    #otherwise check that both previous and next slice have higher dice
    if((next_index in vol_slice_eval) and (prev_index in vol_slice_eval) and (vol_slice_eval[next_index]>curr_val) and (vol_slice_eval[prev_index]>curr_val)):
        return True
    else:
        return False


def get_key_slices_indexes(vol_slice_eval, num_key_images, thresh_value):
    """
    This function gets key images indices based on dice local minimas
    1. A maximum of 5 "bad" images
	2. [One median image for sanity check from the middle] - not implemented
	3. "Bad Image" is defined as local minima (slice after is higher) and below 92 dice
    If there are no "bad images", write 1 worst image
    """
    key_indexes_dict = {}
    sorted_dict = sorted(vol_slice_eval.items(), key=lambda item: item[1])
    curr_iter = iter(sorted_dict)

    num_chosen_images = 0
    while(num_chosen_images< num_key_images):
        try:
            item = curr_iter.__next__()
        except StopIteration:
            break

        index = item[0]
        value = vol_slice_eval[index]
        if(value > thresh_value and key_indexes_dict):#if the smallest value is larger than threshold, return index of the smallest value only
            break
        if(is_local_minimum(vol_slice_eval, index)):
            key_indexes_dict[index] = value
            num_chosen_images = num_chosen_images + 1

    return key_indexes_dict

def overlay_image_mask(img, mask):
    img *= 255.0/img.max()

    img = Image.fromarray(img.astype(np.uint8)).convert("RGBA")
    gt_np = np.zeros([mask.shape[0], mask.shape[1],3], dtype=np.uint8)
    gt_np[:,:,0] = (mask.astype(np.uint8))*255
    gt = Image.fromarray(gt_np).convert("RGBA")
    return Image.blend(img, gt, 0.4)


def resize_image(img,size):

    return img.resize(size, Image.ANTIALIAS)


def save_key_images(key_images_indices, eval_folder, vol_id):
    """
    saves png key images in the evaluation folder
    """
    images_pathes = {}
    print('saving key images for vol: ' + str(vol_id))
    folder_path = os.path.join(eval_folder, str(vol_id))
    truth = nib.load(os.path.join(folder_path, 'truth.nii.gz')).get_data()
    pred = nib.load(os.path.join(folder_path, 'prediction.nii.gz')).get_data()
    volume = nib.load(os.path.join(folder_path, 'data.nii.gz')).get_data()

    key_images_folder = eval_folder + '_key_images/'
    if not os.path.exists(key_images_folder):
        os.makedirs(key_images_folder)

    volume, truth, pred = prepare_for_plotting(volume, truth, pred)
    for key in key_images_indices:
        dice_val = "{0:.2f}".format(key_images_indices[key])
        slice_img = volume[:, :, key - 1]
        truth_img = truth[:, :, key - 1]
        pred_img = pred[:, :, key - 1]
        overlay_truth = overlay_image_mask(slice_img, truth_img)
        overlay_pred = overlay_image_mask(slice_img, pred_img)
        imgs_comb = np.hstack((np.array(overlay_truth), np.array(overlay_pred)))
        res_gt = Image.fromarray(imgs_comb)
        new_size = (512, 256) #specify fixed size for key images
        res_gt = resize_image(res_gt, new_size)

        image_path = key_images_folder + "image_{0}_{1}_{2}.png".format(vol_id, key, dice_val)
        res_gt.save(image_path)
        images_pathes[key] = image_path

    return images_pathes

