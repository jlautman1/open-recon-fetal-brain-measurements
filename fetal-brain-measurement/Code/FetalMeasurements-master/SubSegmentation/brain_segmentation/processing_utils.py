import torch
import numpy as np
import nibabel as nib
from fastai.vision import Image
from scipy import ndimage
from skimage.transform import resize
from scipy.ndimage import zoom
import random
import skimage.exposure

from skimage.morphology import label
from matplotlib import pyplot as plt

ANGLES = [0, 0, 0, 90, 135, 180, 270]
RIGHT_HEMI = 1
CEREBELLUM = 2
RIGHT_LATERAL_VENTRICLE = 3
CSF = 4
LEFT_HEMI = 5
LEFT_LATERAL_VENTRICLE = 6
CLASSES = [RIGHT_HEMI, CEREBELLUM, RIGHT_LATERAL_VENTRICLE, CSF, LEFT_HEMI, LEFT_LATERAL_VENTRICLE]


def _contrast(img, min_range, max_range):
    return skimage.exposure.rescale_intensity(img, in_range=(min_range, max_range), out_range=(0, 1))


def round(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            temp = arr[i][j] % 1
            if temp > 0.5:
                arr[i][j] += 1 - temp
            else:
                arr[i][j] -= temp

    return arr


def normalize(arr):
    minval = arr.min()
    maxval = arr.max()
    if minval != maxval:
        arr -= minval
        arr *= (255.0 / (maxval - minval))
    return arr


def crop_images(images, x_ax, y_ax):
    x_ax_slice = slice(*x_ax)
    y_ax_slice = slice(*y_ax)
    res = []
    for i in range(images.shape[2]):
        imslice = images[:, :, i]
        res.append(imslice[x_ax_slice, y_ax_slice])

    return res


def get_crop_dims(img_data):
    y, x = img_data.T[0].shape
    min_ax = min(x, y)
    x_ax = [0, x]
    y_ax = [0, y]

    i = True
    while x > min_ax:
        x -= 1
        if i:
            x_ax[0] += 1
        else:
            x_ax[1] -= 1
        i = not i

    while y > min_ax:
        y -= 1
        if i:
            y_ax[0] += 1
        else:
            y_ax[1] -= 1
        i = not i

    return x_ax, y_ax


def resize_images(images, x, y, order, preserve_range):
    return [resize(image, (x, y), order=order, preserve_range=preserve_range) for image in images]


def zoom_images(images, order, factor):
    return [zoom(image, zoom=factor, order=order) for image in images]


def update_images(images):
    res = []
    for image in images:
        image = normalize(image)
        image = round(image)
        image /= 255
        res.append(image)

    return res


def uncrop_images(images, zeros, x_ax, y_ax):
    x_ax_slice = slice(*x_ax)
    y_ax_slice = slice(*y_ax)
    res = []
    for image in images:
        cur_zeros = zeros.copy()
        cur_zeros[x_ax_slice, y_ax_slice] = image
        res.append(cur_zeros)

    return res


def save_nifti(images, filename):
    images = np.transpose(images, (1, 2, 0))
    print(images.shape)

    # to save this 3D (ndarry) numpy use this
    ni_img = nib.Nifti1Image(images, affine=np.eye(4))
    ni_img.set_data_dtype(np.float32)
    nib.save(ni_img, filename)
    return ni_img


def _inhomogeneity(img_slice):
    min_val = min(np.unique(img_slice))
    max_val = max(np.unique(img_slice))

    x0 = random.randrange(43, 188)
    y0 = random.randrange(-371, 171)

    zeros = np.zeros_like(img_slice)
    for i in range(img_slice.shape[0]):
        for j in range(img_slice.shape[1]):
            zeros[i, j] = ((x0 + i) ** 2 + (y0 + j) ** 2)

    angle = random.choice([0, 90, -90])
    zeros = ndimage.rotate(zeros, angle, reshape=False)

    img_slice = np.multiply(img_slice, zeros)
    img_slice = np.interp(img_slice, (np.min(np.unique(img_slice)), np.max(np.unique(img_slice))), (min_val, max_val))

    return img_slice


def prepare_for_model(images):
    res = []

    for i, angle in enumerate(ANGLES):
        current_images = []
        for image in images:
            if i in [0, 4]:
                image = _contrast(image, 0.15, 0.85)
            elif i in [1]:
                image = _inhomogeneity(image)

            image = ndimage.rotate(image, angle, reshape=False)
            image = torch.Tensor(image)
            image = image.unsqueeze(0)
            image = Image(image)
            current_images.append(image)

        res.append(current_images)
    return res


def prepare_for_model_no_tta(images):
    res = []
    for image in images:
        image = torch.Tensor(image)
        image = image.unsqueeze(0)
        image = Image(image)
        res.append(image)

    return res


def pre_processing_no_tta(img_data, model_image_size):
    original_x, original_y = img_data[:, :, 0].shape
    min_ax = min(original_y, original_x)
    zeros = np.zeros_like(img_data[:, :, 0])
    x_ax, y_ax = get_crop_dims(img_data)

    images = crop_images(img_data, x_ax, y_ax)
    images = update_images(images)
    images = resize_images(images, model_image_size[0], model_image_size[1], 1, False)
    images = prepare_for_model_no_tta(images)

    return images, min_ax, zeros, x_ax, y_ax


def pre_processing(img_data, model_image_size):
    original_x, original_y = img_data[:, :, 0].shape
    min_ax = min(original_y, original_x)
    zeros = np.zeros_like(img_data[:, :, 0])
    x_ax, y_ax = get_crop_dims(img_data)

    images = crop_images(img_data, x_ax, y_ax)
    images = update_images(images)
    images = resize_images(images, model_image_size[0], model_image_size[1], 1, False)
    images = prepare_for_model(images)

    return images, min_ax, zeros, x_ax, y_ax


def post_processing(images, min_ax, zeros, x_ax, y_ax, filename, model_image_size):
    images = zoom_images(images, 0, min_ax / model_image_size[0])
    images = uncrop_images(images, zeros, x_ax, y_ax)
    images = post_process(np.array(images), "")
    return save_nifti(images, filename)


def majority_vote(rotated_images):
    result_images = rotated_images[0]
    for i, angle in enumerate(ANGLES):
        for j, image in enumerate(rotated_images[i]):
            rotated_images[i][j] = ndimage.rotate(image, -angle, reshape=False)

    for scan_index, image in enumerate(result_images):
        for i in range(160):
            for j in range(160):
                pixel_results = []
                for rot in rotated_images:
                    pixel_seg = rot[scan_index][i][j]
                    if pixel_seg < 0:
                        pixel_seg = 0

                    pixel_results.append(pixel_seg)

                un, cnt = np.unique(pixel_results, return_counts=True)

                most_commons = sorted(un[cnt == cnt.max()])

                if len(most_commons) > 1:
                    if RIGHT_HEMI in most_commons:
                        most_common = RIGHT_HEMI
                    elif LEFT_HEMI in most_commons:
                        most_common = LEFT_HEMI
                    elif CSF in most_commons:
                        most_common = CSF
                    elif CEREBELLUM in most_commons:
                        most_common = CEREBELLUM
                    elif RIGHT_LATERAL_VENTRICLE in most_commons:
                        most_common = RIGHT_LATERAL_VENTRICLE
                    elif LEFT_LATERAL_VENTRICLE in most_commons:
                        most_common = LEFT_LATERAL_VENTRICLE
                    else:
                        most_common = 0
                else:
                    most_common = most_commons[0]

                image[i][j] = most_common

    return result_images


def post_process(pred, filename):
    slices_num = pred.shape[0]
    for i in range(slices_num):
        slice = pred[i, :, :]

        _remove_connected_components_one_class(slice, i, slices_num, pred)
        _remove_connected_components_per_class(slice, i, slices_num, pred)

    return pred


def get_most_common_neighbour(img, i, j):
    result = []
    pixel_value = img[i][j]
    result.append(img[i + 1, j])
    result.append(img[i + 1, j + 1])
    result.append(img[i + 1, j - 1])
    result.append(img[i - 1, j])
    result.append(img[i - 1, j + 1])
    result.append(img[i - 1, j - 1])

    result.append(img[i, j + 1])
    result.append(img[i, j - 1])

    while pixel_value in result:
        result.remove(pixel_value)

    result = result or [0]

    un, cnt = np.unique(result, return_counts=True)

    most_commons = sorted(un[cnt == cnt.max()])
    if len(most_commons) > 1:
        if RIGHT_HEMI in most_commons:
            most_common = RIGHT_HEMI
        elif LEFT_HEMI in most_commons:
            most_common = LEFT_HEMI
        elif CSF in most_commons:
            most_common = CSF
        elif CEREBELLUM in most_commons:
            most_common = CEREBELLUM
        elif RIGHT_LATERAL_VENTRICLE in most_commons:
            most_common = RIGHT_LATERAL_VENTRICLE
        elif LEFT_LATERAL_VENTRICLE in most_commons:
            most_common = LEFT_LATERAL_VENTRICLE
        else:
            most_common = 0
    else:
        most_common = most_commons[0]

    return most_common


def _remove_connected_components_one_class(slice, i, slices_num, pred):
    zeros = np.zeros_like(slice)
    zeros[slice != 0] = 1
    labeled, num = label(zeros, return_num=True, background=0)

    for j in range(1, num + 1):
        indices = np.argwhere(labeled == j)

        if 0 < i < slices_num - 1:
            prev_slice = pred[i - 1, :, :]
            next_slice = pred[i + 1, :, :]

            mask = labeled == j
            prev_overlap = prev_slice[mask] == slice[mask]
            next_overlap = next_slice[mask] == slice[mask]
            next_overlap_ratio = sum(next_overlap) / np.count_nonzero(mask)
            prev_overlap_ratio = sum(prev_overlap) / np.count_nonzero(mask)
            next_is_not_empty = len(np.unique(next_slice)) > 1
            prev_is_not_empty = len(np.unique(prev_slice)) > 1

            if ((next_overlap_ratio < 0.2) and (prev_overlap_ratio < 0.2) and (
                    next_is_not_empty or prev_is_not_empty)):
                pred[i, :, :][indices[:, 0], indices[:, 1]] = 0

        if len(indices) < 15:
            mcn = get_most_common_neighbour(pred[i, :, :], indices[0][0], indices[0][1])
            pred[i, :, :][indices[:, 0], indices[:, 1]] = mcn

def _remove_connected_components_per_class(slice, i, slices_num, pred):
    for c in CLASSES:
        zeros = np.zeros_like(slice)
        zeros[slice == c] = 1

        labeled, num = label(zeros, return_num=True, background=0)

        for j in range(1, num + 1):
            indices = np.argwhere(labeled == j)

            if 0 < i < slices_num - 1:
                prev_slice = pred[i - 1, :, :]
                next_slice = pred[i + 1, :, :]

                mask = labeled == j
                prev_overlap = prev_slice[mask] == slice[mask]
                next_overlap = next_slice[mask] == slice[mask]
                next_overlap_ratio = sum(next_overlap) / np.count_nonzero(mask)
                prev_overlap_ratio = sum(prev_overlap) / np.count_nonzero(mask)
                next_is_not_empty = len(np.unique(next_slice)) > 1
                prev_is_not_empty = len(np.unique(prev_slice)) > 1

                if c in [RIGHT_HEMI, CEREBELLUM, LEFT_HEMI] or (c in [CSF] and (i < 5 or i > slices_num - 5)):
                    if ((next_overlap_ratio < 0.1) and (prev_overlap_ratio < 0.1) and (
                            next_is_not_empty or prev_is_not_empty)):
                        pred[i, :, :][indices[:, 0], indices[:, 1]] = 0

            if c in [LEFT_LATERAL_VENTRICLE, RIGHT_LATERAL_VENTRICLE]:
                if len(indices) < 10:
                    mcn = get_most_common_neighbour(pred[i, :, :], indices[0][0], indices[0][1])
                    pred[i, :, :][indices[:, 0], indices[:, 1]] = mcn
            elif len(indices) < 15:
                mcn = get_most_common_neighbour(pred[i, :, :], indices[0][0], indices[0][1])
                pred[i, :, :][indices[:, 0], indices[:, 1]] = mcn


def acc_no_bg(input, target):
    target = target.squeeze(1)
    mask = target != 0
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()