import argparse
import glob
import json
import os
from pathlib import Path
import numpy as np
from scipy import ndimage
from fetal_segmentation.data_curation.helper_functions import move_smallest_axis_to_z, patient_id_from_filepath, origin_id_from_filepath, swap_to_original_axis
from fetal_segmentation.data_generation.cut_relevant_areas import find_bounding_box, check_bounding_box
from fetal_segmentation.data_generation.preprocess import window_1_99, normalize_data
from fetal_segmentation.evaluation.eval_utils.postprocess import postprocess_prediction
from fetal_segmentation.evaluation.eval_utils.prediction import patch_wise_prediction, predict_augment, predict_flips
from fetal_segmentation.training.train_functions.training import load_old_model, get_last_model_path
from fetal_segmentation.utils.read_write_data import list_load
from fetal_segmentation.utils.read_write_data import save_nifti, read_img
import fetal_segmentation.data_generation.preprocess
import nibabel as nib


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def secondary_prediction(mask, vol, config2, model2,
                         preprocess_method2=None, norm_params2=None,
                         overlap_factor=0.9, augment2=None, num_augment=10, return_all_preds=False):

    pred = mask
    bbox_start, bbox_end = find_bounding_box(pred)
    check_bounding_box(pred, bbox_start, bbox_end)
    padding = np.array([16, 16, 8])

    bbox_start = np.maximum(bbox_start - padding, 0)
    bbox_end = np.minimum(bbox_end + padding, mask.shape)

    print(f"🧠 bbox_start: {bbox_start}, bbox_end: {bbox_end}")
    if np.any(bbox_start >= bbox_end):
        raise ValueError(f"Invalid bounding box: {bbox_start} to {bbox_end}")

    roi = mask[bbox_start[0]:bbox_end[0],
               bbox_start[1]:bbox_end[1],
               bbox_start[2]:bbox_end[2]]

    data = vol[bbox_start[0]:bbox_end[0],
               bbox_start[1]:bbox_end[1],
               bbox_start[2]:bbox_end[2]]
    #nib.save(nib.Nifti1Image(data, np.eye(4)), "./fetal-brain-measurement/output/Pat172_Se12_Res0.7813_0.7813_Spac3.0/debug_secondary_input_patch.nii.gz")

    data = preproc_and_norm(data, preprocess_method2, norm_params2)

    prediction = get_prediction(data, model2,
                                 augment=augment2,
                                 num_augments=num_augment,
                                 return_all_preds=return_all_preds,
                                 overlap_factor=overlap_factor,
                                 config=config2)

    # ⚠️ OLD: padding prediction into full volume
    # padding2 = list(zip(bbox_start, np.array(vol.shape) - bbox_end))
    # prediction = np.pad(prediction, padding2, mode='constant', constant_values=0)

    # Convert to binary BEFORE inserting
    binary_prediction = postprocess_prediction(prediction, threshold=0.5)

    # Insert into full volume
    full_prediction = np.zeros_like(mask, dtype=np.uint8)
    full_prediction[
        bbox_start[0]:bbox_end[0],
        bbox_start[1]:bbox_end[1],
        bbox_start[2]:bbox_end[2]
    ] = binary_prediction.astype(np.uint8)

    # # Save debug cropped prediction
    # nib.save(nib.Nifti1Image(binary_prediction.astype(np.uint8), np.eye(4)), 
    #         "./fetal-brain-measurement/output/Pat172_Se12_Res0.7813_0.7813_Spac3.0/debug_pred_roi_bin.nii.gz")

    # # Save debug full prediction
    # nib.save(nib.Nifti1Image(full_prediction.astype(np.uint8), np.eye(4)), 
    #         "./fetal-brain-measurement/output/Pat172_Se12_Res0.7813_0.7813_Spac3.0/debug_full_secondary.nii.gz")

    return full_prediction



def preproc_and_norm(data, preprocess_method=None, norm_params=None, scale=None, preproc=None):
    if preprocess_method is not None:
        print('Applying preprocess by {}...'.format(preprocess_method))
        if preprocess_method == 'window_1_99':
            data = window_1_99(data)
        else:
            raise Exception('Unknown preprocess: {}'.format(preprocess_method))

    if scale is not None:
        data = ndimage.zoom(data, scale)
    if preproc is not None:
        preproc_func = getattr(data_generation.preprocess, preproc)
        data = preproc_func(data)

    # data = normalize_data(data, mean=data.mean(), std=data.std())
    if norm_params is not None and any(norm_params.values()):
        data = normalize_data(data, mean=norm_params['mean'], std=norm_params['std'])
    return data


def get_prediction(data, model, augment, num_augments, return_all_preds, overlap_factor, config):
    if augment is not None:
        patch_shape = config["patch_shape"] + [config["patch_depth"]]
        if augment == 'all':
            prediction = predict_augment(data, model=model, overlap_factor=overlap_factor, num_augments=num_augments, patch_shape=patch_shape)
        elif augment == 'flip':
            prediction = predict_flips(data, model=model, overlap_factor=overlap_factor, patch_shape=patch_shape, config=config)
        else:
            raise ("Unknown augmentation {}".format(augment))
        if not return_all_preds:
            prediction = np.median(prediction, axis=0)
    else:
        prediction = \
            patch_wise_prediction(model=model, 
                                  data=np.expand_dims(data, 0),
                                #   data = np.expand_dims(data.cpu().numpy(), 0),   # move to CPU, then NumPy
                                #  data = np.expand_dims(data.detach().cpu().numpy(), 0),
                                  overlap_factor=overlap_factor,
                                  patch_shape=config["patch_shape"] + [config["patch_depth"]])
    prediction = prediction.squeeze()
    return prediction

def delete_nii_gz(s):
    if s[-3:] == '.gz':
        s = s[:-3]
    if s[-4:] == '.nii':
        s = s[:-4]
    return s


def main(input_path, output_path, has_gt, scan_id, overlap_factor,
         config, model, preprocess_method=None, norm_params=None, augment=None, num_augment=0,
         config2=None, model2=None, preprocess_method2=None, norm_params2=None, augment2=None, num_augment2=0,
         z_scale=None, xy_scale=None, return_all_preds=False):

    output_path = output_path + 'test/' + str(scan_id) + '/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    

    print('Loading nifti from {}...'.format(input_path))
    nifti_data = read_img(input_path).get_fdata()
    print('Predicting mask...')
    save_nifti(nifti_data, os.path.join(output_path,  'data.nii.gz'))
    nifti_data, swap_axis = move_smallest_axis_to_z(nifti_data)
    data_size = nifti_data.shape
    data = nifti_data.astype(np.float64).squeeze()
    print('original_shape: ' + str(data.shape))

    if (z_scale is None):
        z_scale = 1.0
    if (xy_scale is None):
        xy_scale = 1.0
    if z_scale != 1.0 or xy_scale != 1.0:
        data = ndimage.zoom(data, [xy_scale, xy_scale, z_scale])

    data = preproc_and_norm(data, preprocess_method, norm_params,
                            scale=config.get('scale_data', None),
                            preproc=config.get('preproc', None))

 #   data = np.pad(data, 3, 'constant', constant_values=data.min())
    print("case: " + str(scan_id))
    print('Shape: ' + str(data.shape))
    prediction = get_prediction(data=data, model=model, augment=augment,
                                num_augments=num_augment, return_all_preds=return_all_preds,
                                overlap_factor=overlap_factor, config=config)
    # unpad
 #   prediction = prediction[3:-3, 3:-3, 3:-3]


    # revert to original size
    if config.get('scale_data', None) is not None:
        prediction = ndimage.zoom(prediction.squeeze(), np.divide([1, 1, 1], config.get('scale_data', None)), order=0)[..., np.newaxis]

    if z_scale != 1.0 or xy_scale != 1.0:
        prediction = prediction.squeeze()
        prediction = ndimage.zoom(prediction, [data_size[0]/prediction.shape[0], data_size[1]/prediction.shape[1], data_size[2]/prediction.shape[2]], order=1)[..., np.newaxis]

    prediction = prediction.squeeze()

    mask = postprocess_prediction(prediction, threshold=0.5)


    if config2 is not None:
        swapped_mask = swap_to_original_axis(swap_axis, mask)
        save_nifti(np.int16(swapped_mask), os.path.join(output_path, 'prediction_all.nii.gz'))
        prediction = secondary_prediction(mask, vol=nifti_data.astype(np.float64),
                                          config2=config2, model2=model2,
                                          preprocess_method2=preprocess_method2, norm_params2=norm_params2,
                                          overlap_factor=overlap_factor, augment2=augment2, num_augment=num_augment2,
                                          return_all_preds=return_all_preds)

        prediction_binarized = postprocess_prediction(prediction,  threshold=0.5)

        if(return_all_preds):
            prediction = swap_to_original_axis(swap_axis, prediction)
            save_nifti(prediction, os.path.join(output_path, 'predictions'+'.nii.gz'))
            for i in range(len(prediction_binarized)):
                prediction_binarized[i] = swap_to_original_axis(swap_axis, prediction_binarized[i])
                save_nifti(np.int16(prediction_binarized[i]), os.path.join(output_path, 'tta'+str(i)+'.nii.gz'))
        else:
            prediction_binarized = swap_to_original_axis(swap_axis, prediction_binarized)
            save_nifti(np.int16(prediction_binarized), os.path.join(output_path, 'prediction.nii.gz'))

    else: #if there is no secondary prediction, save the first network prediction or predictions as the final ones
        if(return_all_preds):
            prediction = swap_to_original_axis(swap_axis, prediction)
            save_nifti(prediction, os.path.join(output_path, 'predictions'+'.nii.gz'))
            for i in range(len(mask)):
                mask[i] = swap_to_original_axis(swap_axis, mask[i])
                save_nifti(np.int16(mask[i]), os.path.join(output_path, 'tta'+str(i)+'.nii.gz'))
        else:
            mask = swap_to_original_axis(swap_axis, mask)
            save_nifti(np.int16(mask), os.path.join(output_path, 'prediction.nii.gz'))

    if(has_gt):
        volume_dir = os.path.dirname(input_path)
        gt_path = os.path.join(volume_dir,'truth.nii')
        if(not os.path.exists(gt_path)):
            gt_path = os.path.join(volume_dir,'truth.nii.gz')
        truth = read_img(gt_path).get_fdata()
        save_nifti(np.int16(truth), os.path.join(output_path, 'truth.nii.gz'))

    print('Saving to {}'.format(output_path))
    print('Finished.')


def get_params(config_dir):
    with open(os.path.join(config_dir, 'config.json'), 'r') as f:
        __config = json.load(f)
    with open(os.path.join(config_dir, 'norm_params.json'), 'r') as f:
        __norm_params = json.load(f)
    __model_path = config_dir  # Correctly use the given directory

    return __config, __norm_params, __model_path




def predict_single_case(volume_path):
    scan_id = origin_id_from_filepath(volume_path)
    print('input path is: ' + volume_path)
    main(volume_path, opts.output_folder, is_labeled, scan_id,  overlap_factor=opts.overlap_factor,
         config=_config, model=_model, preprocess_method=opts.preprocess, norm_params=_norm_params, augment=opts.augment,
         num_augment=opts.num_augment,
         config2=_config2, model2=_model2, preprocess_method2=opts.preprocess2, norm_params2=_norm_params2, augment2=opts.augment2,
         num_augment2=opts.num_augment2,
         z_scale=opts.z_scale, xy_scale=opts.xy_scale, return_all_preds=opts.return_all_preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="specifies nifti file dir path or filename in case of single prediction",
                        type=str, required=True)
    parser.add_argument("--ids_list", help="specifies which scans from the directory to use for inference. "
                                           "By default, all scans from the directory are used. Expected to be in config_dir",
                        type=str, required=False)
    parser.add_argument("--output_folder", help="specifies nifti file dir path",
                        type=str, required=True)
    parser.add_argument("--overlap_factor", help="specifies overlap between prediction patches",
                        type=float, default=0.7)
    parser.add_argument("--z_scale", help="specifies overlap between prediction patches",
                        type=float, default=1)
    parser.add_argument("--xy_scale", help="specifies overlap between prediction patches",
                        type=float, default=1)
    parser.add_argument("--return_all_preds", help="return all predictions or mean result for prediction?",
                        type=int, default=0)
    parser.add_argument("--labeled", help="in case of labeled data, copy ground truth for convenience",
                        type=str2bool, default=False)
    parser.add_argument("--all_in_one_dir", help="in case of unlabeled data, this option allows to have all the volumes in one directory without directory hierarchy",
                        type=str2bool, default=False)
    parser.add_argument("--predict_single", help="in case of unlabeled data, this option allows to have all the volumes in one directory without directory hierarchy",
                        type=str2bool, default=False)
    # Params for primary prediction
    parser.add_argument("--config_dir", help="specifies config dir path",
                        type=str, required=True)
    parser.add_argument("--preprocess", help="which preprocess to do",   #currently, only 'window_1_99' preprocess is supported, better to use it via preprocess parameter
                                                                        #in config file. Use only if it wasn't specified in config file
                        type=str, required=False, default=None)
    parser.add_argument("--augment", help="what augment to do",
                        type=str, required=False, default=None)  # one of 'flip, all'
    parser.add_argument("--num_augment", help="what augment to do",
                        type=int, required=False, default=0)  # one of 'flip, all'

    # Params for secondary prediction
    parser.add_argument("--config2_dir", help="specifies config dir path",
                        type=str, required=False, default=None)
    parser.add_argument("--preprocess2", help="what preprocess to do",
                        type=str, required=False, default=None)
    parser.add_argument("--augment2", help="what augment to do",
                        type=str, required=False, default=None)  # one of 'flip, all'
    parser.add_argument("--num_augment2", help="what augment to do",
                        type=int, required=False, default=0)  # one of 'flip, all'

    opts = parser.parse_args()

    print(opts.input_path)
    is_labeled = opts.labeled

    Path(opts.output_folder).mkdir(exist_ok=True)

    # 1
    _config, _norm_params, _model_path = get_params(opts.config_dir)

    # 2
    if opts.config2_dir is not None:
        _config2, _norm_params2, _model2_path = get_params(opts.config2_dir)
    else:
        _config2, _norm_params2, _model2_path = None, None, None

    if(opts.ids_list != None):
        scans_list = list_load(os.path.join(opts.config_dir, opts.ids_list))
        print(*scans_list)

    print('First:' + _model_path)
    _model = load_old_model(get_last_model_path(_model_path), config=_config)

    if(_model2_path is not None):
        print('Second:' + _model2_path)
        _model2 = load_old_model(get_last_model_path(_model2_path), config=_config2)
    else: _model2 = None

    all_in_one_dir = opts.all_in_one_dir #is the data arranged in directory hierarchy or all volumes in one dir?
    predict_single = opts.predict_single

    if(all_in_one_dir):
        for volume_path in glob.glob(os.path.join(opts.input_path, '*')):
            scan_id = origin_id_from_filepath(volume_path)
            patient_id = patient_id_from_filepath(volume_path)
            if (opts.ids_list != None) and (patient_id in scans_list):#use only scans that are specified in ids_list
                continue
            print('input path is: ' + volume_path)
            main(volume_path, opts.output_folder, is_labeled, scan_id,  overlap_factor=opts.overlap_factor,
                 config=_config, model=_model, preprocess_method=opts.preprocess, norm_params=_norm_params, augment=opts.augment,
                 num_augment=opts.num_augment,
                 config2=_config2, model2=_model2, preprocess_method2=opts.preprocess2, norm_params2=_norm_params2, augment2=opts.augment2,
                 num_augment2=opts.num_augment2,
                 z_scale=opts.z_scale, xy_scale=opts.xy_scale, return_all_preds=opts.return_all_preds)
    elif(predict_single):
        filepath = opts.input_path
        predict_single_case(filepath)
    else:
        scan_dirs = os.listdir(opts.input_path)
        for dir in scan_dirs:
            scan_id = dir
            if (opts.ids_list != None) and (scan_id not in scans_list):#use only scans that are specified in ids_list
                continue
            volume_path = opts.volume_path + dir + '/'
            volume_path = volume_path + 'volume.nii'
            if(not os.path.exists(volume_path)):
                volume_path = volume_path + 'volume.nii.gz'
            if(not os.path.exists(volume_path)):
                filenames = glob.glob(volume_path + '/Pat*.nii')
                volume_path = filenames[0]
            print('input path is: ' + volume_path)
            main(volume_path, opts.output_folder, is_labeled, scan_id,  overlap_factor=opts.overlap_factor,
                 config=_config, model=_model, preprocess_method=opts.preprocess, norm_params=_norm_params, augment=opts.augment,
                 num_augment=opts.num_augment,
                 config2=_config2, model2=_model2, preprocess_method2=opts.preprocess2, norm_params2=_norm_params2, augment2=opts.augment2,
                 num_augment2=opts.num_augment2,
                 z_scale=opts.z_scale, xy_scale=opts.xy_scale, return_all_preds=opts.return_all_preds)

