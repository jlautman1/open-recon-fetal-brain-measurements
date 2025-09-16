import itertools
import json
import os

import nibabel as nib
import pandas as pd
import tables
from keras import Model
from scipy import ndimage
from tqdm import tqdm

from fetal_segmentation.data_generation.augment import contrast_augment
from fetal_segmentation.data_generation.preprocess import window_1_99, normalize_data
from fetal_segmentation.evaluation.eval_utils.eval_functions import dice_coefficient, calc_dice_per_slice
from fetal_segmentation.evaluation.eval_utils.postprocess import *
from fetal_segmentation.training.train_functions.training import load_old_model, get_last_model_path
from fetal_segmentation.utils.read_write_data import pickle_load, load_nifti
from fetal_segmentation.utils.threaded_generator import ThreadedGenerator


def get_set_of_patch_indices_full(start, stop, step):
    indices = []
    for start_i, stop_i, step_i in zip(start, stop, step):
        indices_i = list(range(start_i, stop_i + 1, step_i))
        if stop_i % step_i > 0:
            indices_i += [stop_i]
        indices += [indices_i]
    return np.array(list(itertools.product(*indices)))


def fix_out_of_bound_patch_attempt(data, patch_shape, patch_index, ndim=3):
    """
    Pads the data and alters the patch index so that a patch will be correct.
    :param data:
    :param patch_shape:
    :param patch_index:
    :return: padded data, fixed patch index
    """
    image_shape = data.shape[-ndim:]
    pad_before = np.abs((patch_index < 0) * patch_index)
    pad_after = np.abs(((patch_index + patch_shape) > image_shape) * ((patch_index + patch_shape) - image_shape))
    pad_args = np.stack([pad_before, pad_after], axis=1)
    if pad_args.shape[0] < len(data.shape):
        pad_args = [[0, 0]] * (len(data.shape) - pad_args.shape[0]) + pad_args.tolist()
    data = np.pad(data, pad_args, mode="edge")
    patch_index += pad_before
    return data, patch_index

def get_patch_from_3d_data(data, patch_shape, patch_index):
    """
    Returns a patch from a numpy array.
    :param data: numpy array from which to get the patch.
    :param patch_shape: shape/size of the patch.
    :param patch_index: corner index of the patch.
    :return: numpy array take from the data with the patch shape specified.
    """
    patch_index = np.asarray(patch_index, dtype=np.int16)
    patch_shape = np.asarray(patch_shape)
    image_shape = data.shape[-3:]
    if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
        data, patch_index = fix_out_of_bound_patch_attempt(data, patch_shape, patch_index)
    return data[..., patch_index[0]:patch_index[0] + patch_shape[0],
                     patch_index[1]:patch_index[1] + patch_shape[1],
                     patch_index[2]:patch_index[2] + patch_shape[2]]

def batch_iterator(indices, batch_size, data_0, patch_shape, truth_0, prev_truth_index, truth_patch_shape):
    i = 0
    while i < len(indices):
        batch = []
        curr_indices = []
        while len(batch) < batch_size and i < len(indices):
            curr_index = indices[i]
            patch = get_patch_from_3d_data(data_0, patch_shape=patch_shape, patch_index=curr_index)
            if truth_0 is not None:
                truth_index = list(curr_index[:2]) + [curr_index[2] + prev_truth_index]
                truth_patch = get_patch_from_3d_data(truth_0, patch_shape=truth_patch_shape,
                                                     patch_index=truth_index)
                patch = np.concatenate([patch, truth_patch], axis=-1)
            batch.append(patch)
            curr_indices.append(curr_index)
            i += 1
        yield [batch, curr_indices]

def predict(model, data):
    return model.predict(data)

def patch_wise_prediction(model: Model, data, patch_shape, overlap_factor=0, batch_size=5,
                          permute=False, truth_data=None, prev_truth_index=None, prev_truth_size=None):
    """
    :param truth_data:
    :param permute:
    :param overlap_factor:
    :param batch_size:
    :param model:
    :param data:
    :return:
    """
    is3d = np.sum(np.array(model.output_shape[1:]) > 1) > 2

    if is3d:
        prediction_shape = model.output_shape[-3:]
    else:
        prediction_shape = model.output_shape[-3:-1] + (1,)  # patch_shape[-3:-1] #[64,64]#
    min_overlap = np.subtract(patch_shape, prediction_shape)
    max_overlap = np.subtract(patch_shape, (1, 1, 1))
    overlap = min_overlap + (overlap_factor * (max_overlap - min_overlap)).astype(int)
    #overlap = min_overlap + (overlap_factor * (max_overlap - min_overlap)).astype(np.int)
    data_0 = np.pad(data[0],
                    [(np.ceil(_ / 2).astype(int), np.floor(_ / 2).astype(int)) for _ in
                     np.subtract(patch_shape, prediction_shape)],
                    mode='constant', constant_values=np.percentile(data[0], q=1))
    pad_for_fit = [(np.ceil(_ / 2).astype(int), np.floor(_ / 2).astype(int)) for _ in
                   np.maximum(np.subtract(patch_shape, data_0.shape), 0)]
    data_0 = np.pad(data_0,
                    [_ for _ in pad_for_fit],
                    'constant', constant_values=np.percentile(data_0, q=1))

    if truth_data is not None:
        truth_0 = np.pad(truth_data[0],
                         [(np.ceil(_ / 2).astype(int), np.floor(_ / 2).astype(int)) for _ in
                          np.subtract(patch_shape, prediction_shape)],
                         mode='constant', constant_values=0)
        truth_0 = np.pad(truth_0, [_ for _ in pad_for_fit],
                         'constant', constant_values=0)

        truth_patch_shape = list(patch_shape[:2]) + [prev_truth_size]
    else:
        truth_0 = None
        truth_patch_shape = None

    indices = get_set_of_patch_indices_full((0, 0, 0),
                                            np.subtract(data_0.shape, patch_shape),
                                            np.subtract(patch_shape, overlap))

    b_iter = batch_iterator(indices, batch_size, data_0, patch_shape,
                            truth_0, prev_truth_index, truth_patch_shape)
    tb_iter = iter(ThreadedGenerator(b_iter, queue_maxsize=50))

    data_shape = list(data.shape[-3:] + np.sum(pad_for_fit, -1))
    if is3d:
        data_shape += [model.output_shape[1]]
    else:
        data_shape += [model.output_shape[-1]]
    predicted_output = np.zeros(data_shape)
    predicted_count = np.zeros(data_shape, dtype=np.int16)
    with tqdm(total=len(indices)) as pbar:
        for [curr_batch, batch_indices] in tb_iter:
            curr_batch = np.asarray(curr_batch)
            if is3d:
                curr_batch = np.expand_dims(curr_batch, 1)
            prediction = model.predict(curr_batch)

            if is3d:
                prediction = prediction.transpose([0, 2, 3, 4, 1])
            else:
                prediction = np.expand_dims(prediction, -2)

            for predicted_patch, predicted_index in zip(prediction, batch_indices):
                # predictions.append(predicted_patch)
                x, y, z = predicted_index
                x_len, y_len, z_len = predicted_patch.shape[:-1]
                predicted_output[x:x + x_len, y:y + y_len, z:z + z_len, :] += predicted_patch
                predicted_count[x:x + x_len, y:y + y_len, z:z + z_len] += 1
            pbar.update(batch_size)

    assert np.all(predicted_count > 0), 'Found zeros in count'

    if np.sum(pad_for_fit) > 0:
        # must be a better way :\
        x_pad, y_pad, z_pad = [[None if p2[0] == 0 else p2[0],
                                None if p2[1] == 0 else -p2[1]] for p2 in pad_for_fit]
        predicted_count = predicted_count[x_pad[0]: x_pad[1],
                          y_pad[0]: y_pad[1],
                          z_pad[0]: z_pad[1]]
        predicted_output = predicted_output[x_pad[0]: x_pad[1],
                           y_pad[0]: y_pad[1],
                           z_pad[0]: z_pad[1]]

    assert np.array_equal(predicted_count.shape[:-1], data[0].shape), 'prediction shape wrong'
    return predicted_output / predicted_count

def predict_flips(data, model, overlap_factor, config):
    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(0, len(s) + 1))

    def predict_it(data_, axes=()):
        data_ = flip_it(data_, axes)
        curr_pred = \
            patch_wise_prediction(model=model,
                                  data=np.expand_dims(data_.squeeze(), 0),
                                  overlap_factor=overlap_factor,
                                  patch_shape=config["patch_shape"] + [config["patch_depth"]]).squeeze()
        curr_pred = flip_it(curr_pred, axes)
        return curr_pred

    predictions = []
    for axes in powerset([0, 1, 2]):
        predictions += [predict_it(data, axes).squeeze()]

    return predictions


def flip_it(data_, axes):
    for ax in axes:
        data_ = np.flip(data_, ax)
    return data_


def predict_augment(data, model, overlap_factor, patch_shape, num_augments=32):
    data_max = data.max()
    data_min = data.min()
    data = data.squeeze()

    order = 2
    predictions = []
    for _ in range(num_augments):
        # pixel-wise augmentations
        #contrast augmentation
        val_range = data_max - data_min
        contrast_min_val = data_min + 0.10 * np.random.uniform(-1, 1) * val_range
        contrast_max_val = data_max + 0.10 * np.random.uniform(-1, 1) * val_range
        curr_data = contrast_augment(data, contrast_min_val, contrast_max_val)

        # spatial augmentations
        #rotate, flip, transpose

        rotate_factor = np.random.uniform(-30, 30)
        to_flip = np.arange(0, 3)[np.random.choice([True, False], size=3)]
        to_transpose = np.random.choice([True, False])

        curr_data = flip_it(curr_data, to_flip)

        if to_transpose:
            curr_data = curr_data.transpose([1, 0, 2])

        curr_data = ndimage.rotate(curr_data, rotate_factor, order=order, reshape=False)

        curr_prediction = patch_wise_prediction(model=model, data=curr_data[np.newaxis, ...], overlap_factor=overlap_factor, patch_shape=patch_shape).squeeze()

        curr_prediction = ndimage.rotate(curr_prediction, -rotate_factor, reshape=False)

        if to_transpose:
            curr_prediction = curr_prediction.transpose([1, 0, 2])

        curr_prediction = flip_it(curr_prediction, to_flip)
        predictions += [curr_prediction.squeeze()]

    res = np.stack(predictions, axis=0)
    return res


def preproc_and_norm(data, preprocess_method, norm_params):
    if preprocess_method is not None:
        print('Applying preprocess by {}...'.format(preprocess_method))
        if preprocess_method == 'window_1_99':
            data = window_1_99(data)
        else:
            raise Exception('Unknown preprocess: {}'.format(preprocess_method))

    if norm_params is not None and any(norm_params.values()):
        data = normalize_data(data, mean=norm_params['mean'], std=norm_params['std'])
    return data

def predict_case(model, input_path, patch_shape, patch_depth, preprocess, norm_params, overlap_factor, gt_path=None):

    print('Loading mat from {}...'.format(input_path))
    nifti = load_nifti(input_path)
    print('Predicting mask...')
    data = nifti.get_fdata().astype(np.float)

    data = preproc_and_norm(data, preprocess, norm_params)

    prediction = \
        patch_wise_prediction(model=model,
                              data=np.expand_dims(data, 0),
                              overlap_factor=overlap_factor,
                              patch_shape=patch_shape + [patch_depth])

    print('Post-processing mask...')
    if prediction.shape[-1] > 1:
        prediction = prediction[..., 1]
    prediction = prediction.squeeze()
    print("Storing prediction in [7-9], 7 should be the best...")
    mask = postprocess_prediction(prediction, threshold=0.5)

    return mask, nifti

def get_prediction_params(config_dir, normalize=True):
    with open(os.path.join(config_dir, 'config.json'), 'r') as f:
        __config = json.load(f)
    __norm_params = None
    if(normalize):
        with open(os.path.join(config_dir, 'norm_params.json'), 'r') as f:
            __norm_params = json.load(f)
    __model_path = os.path.join(config_dir, (__config['model_pref']))
    return __config, __norm_params, __model_path


def save_case_data(output_dir,case_ind, dice_per_slice_dict, nifti_vol, nifti_gt, prediction):

    case_out_directory = os.path.join(output_dir, case_ind)
    if not os.path.exists(case_out_directory):
        os.makedirs(case_out_directory)

    df_scan = pd.DataFrame(dice_per_slice_dict, index=[0])
    df_scan.to_csv(os.path.join(case_out_directory,  'dice_per_slice.csv'))

    nib.save(nifti_vol, os.path.join(case_out_directory,  'vol.nii'))
    nib.save(nifti_gt, os.path.join(case_out_directory, 'gt.nii'))

    nifti_prediction = nib.nifti1.Nifti1Image(prediction, nifti_vol.affine, nifti_vol.header)
    nib.save(nifti_prediction, os.path.join(case_out_directory, 'pred.nii'))


def evaluate_cases(validation_keys_file, model_file, hdf5_file, patch_shape, patch_depth, output_dir, raw_data_path, preprocess, norm_params, overlap_factor):
    file_names = []
    dice_dict = {}

    model = load_old_model(get_last_model_path(model_file))
    validation_indices = pickle_load(validation_keys_file)
    data_file = tables.open_file(hdf5_file, "r")
    for index in validation_indices:
        case_ind = data_file.root.subject_ids[index].decode('utf-8')
        if(case_ind=="46"): #remove this!!!!
            continue

        vol_path = raw_data_path + '/' + case_ind + '/volume.nii'
        gt_path = raw_data_path + '/' + case_ind + '/truth.nii'

        prediction, nifti_vol = predict_case(model, vol_path, patch_shape, patch_depth, preprocess, norm_params, overlap_factor)
        nifti_gt = load_nifti(gt_path)
        numpy_gt = nifti_gt.get_fdata().astype(np.float)

        vol_dice = dice_coefficient(numpy_gt, prediction)
        dice_dict[case_ind] = vol_dice
        dice_per_slice_dict = calc_dice_per_slice(numpy_gt, prediction)


        save_case_data(output_dir,case_ind, dice_per_slice_dict, nifti_vol, nifti_gt, prediction)

    data_file.close()
    df_total = pd.DataFrame(dice_dict, index=[0])
    df_total.to_csv(os.path.join(output_dir,  'volume_dices.csv'))

    return file_names