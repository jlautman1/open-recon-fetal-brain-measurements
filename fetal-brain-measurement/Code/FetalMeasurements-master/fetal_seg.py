import os
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
import torch
import sys
import nibabel as nib


sys.path.append("./fetal_segmentation")

from fetal_segmentation.evaluation.predict_nifti_dir import get_params, preproc_and_norm, \
    get_prediction, secondary_prediction
from fetal_segmentation.training.train_functions.training import load_old_model, get_last_model_path
from fetal_segmentation.utils.read_write_data import save_nifti, read_img, save_nifti_pred
from fetal_segmentation.data_curation.helper_functions import move_smallest_axis_to_z, \
    swap_to_original_axis
from fetal_segmentation.evaluation.eval_utils.postprocess import postprocess_prediction



class FetalSegmentation(object):
    def __init__(self, config_roi_dir, model_roi,
                 config_secondnet_dir, model_net,
                 ):
        
        self._config, self._norm_params, self._model_path = get_params(config_roi_dir)

        # LOad second network if possible
        if config_secondnet_dir is not None:
            self._config2, self._norm_params2, self._model2_path = get_params(config_secondnet_dir)
        else:
            self._config2, self._norm_params2, self._model2_path = None, None, None

        print('First:' + self._model_path)
        print("DEBUG model path =", get_last_model_path(self._model_path))
        self._model = load_old_model(get_last_model_path(self._model_path), config=self._config)

        if (self._model2_path is not None):
            print('Second:' + self._model2_path)
            self._model2 = load_old_model(get_last_model_path(self._model2_path), config=self._config2)
        else:
            self._model2 = None

    def predict(self, in_file, output_path,
                overlap_factor=0.7,
                z_scale=None, xy_scale=None):
        # def main(input_path, output_path, has_gt, scan_id, overlap_factor,
        #      config, model, preprocess_method=None, norm_params=None, augment=None, num_augment=0,
        #      config2=None, model2=None, preprocess_method2=None, norm_params2=None, augment2=None, num_augment2=0,
        #      z_scale=None, xy_scale=None, return_all_preds=False):

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        print('Loading nifti from {}...'.format(in_file))
        nifti_data = read_img(in_file).get_fdata()
        print(f"↳ Loaded image shape: {nifti_data.shape}, dtype: {nifti_data.dtype}")
        print(f"↳ Min/Max: {nifti_data.min()} / {nifti_data.max()}")
        print('Predicting mask...')
        save_nifti(nifti_data, os.path.join(output_path, 'data.nii.gz'))
        print("🔁 Moving smallest axis to Z...")
        nifti_data, swap_axis = move_smallest_axis_to_z(nifti_data)
        print(f"↳ New shape: {nifti_data.shape}, Swap axis: {swap_axis}")
        data_size = nifti_data.shape
        data = nifti_data.astype(float).squeeze()
        print(f"🔎 After squeeze, shape: {data.shape}")

        if (z_scale is None):
            print("z_scale is None")
            z_scale = 1.0
        if (xy_scale is None):
            print("xy_scale is None")
            xy_scale = 1.0
        if z_scale != 1.0 or xy_scale != 1.0:
            print("z_scale != 1.0 or xy_scale != 1.0")
            data = ndimage.zoom(data, [xy_scale, xy_scale, z_scale])
        print(f"🧮 Scale: z_scale = {z_scale}, xy_scale = {xy_scale}")
        print(f"↳ New shape after zoom: {data.shape}")


        #data = preproc_and_norm(data, preprocess_method="window_1_99", norm_params=self._norm_params,
        #                        scale=self._config.get('scale_data', None),
         #                       preproc=self._config.get('preproc', None))

        #print('Shape: ' + str(data.shape))
        #prediction = get_prediction(data=data, model=self._model, augment=None,
         #                           num_augments=1, return_all_preds=None,
          #                          overlap_factor=overlap_factor, config=self._config)
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("before preproc - min:", np.min(data), "max:", np.max(data), "mean:", np.mean(data), "std:", np.std(data))
        #print("starting preproc and norm, data right now:", data)
        data = preproc_and_norm(data, preprocess_method="window_1_99", norm_params=self._norm_params,
                                scale=self._config.get('scale_data', None),
                                preproc=self._config.get('preproc', None))

        # # Explicitly sending data tensor to GPU
        # data_tensor = torch.from_numpy(data).float().to(device)

        # prediction = get_prediction(data=data_tensor, model=self._model, augment=None,
        #                             num_augments=1, return_all_preds=None,
        #                             overlap_factor=overlap_factor, config=self._config)
        #print("after preproc before get prediction, data: ", data)
        print("after preproc - min:", np.min(data), "max:", np.max(data), "mean:", np.mean(data), "std:", np.std(data))
        prediction = get_prediction(data=data,
                                   model=self._model,
                                   augment=None,
                                   num_augments=1,
                                   return_all_preds=False,
                                   overlap_factor=overlap_factor,
                                   config=self._config)
        
        print(f"✅ Prediction done. Shape: {prediction.shape}, dtype: {prediction.dtype}")
        print(f"↳ Prediction stats: min = {np.min(prediction)}, max = {np.max(prediction)}")

        #print("revert to original size, prediction: ", prediction)
        # revert to original size
        if self._config.get('scale_data', None) is not None:
            print("🔁 Resizing prediction back (undoing scale_data)...:", self._config.get('scale_data', None))
            prediction = \
                ndimage.zoom(prediction.squeeze(), np.divide([1, 1, 1], self._config.get('scale_data', None)), order=0)[
                    ..., np.newaxis]

        print(f"↳ Shape after unscale: {prediction.shape}")

        if z_scale != 1.0 or xy_scale != 1.0:
            print("🔁 Resizing prediction to match original input shape...")
            prediction = prediction.squeeze()
            prediction = ndimage.zoom(prediction,
                                      [data_size[0] / prediction.shape[0], data_size[1] / prediction.shape[1],
                                       data_size[2] / prediction.shape[2]], order=1)[..., np.newaxis]
            print(f"↳ Final prediction shape: {prediction.shape}")
        prediction = prediction.squeeze()
        print("🧽 Binarizing with postprocess_prediction (threshold=0.5)...")
        mask = postprocess_prediction(prediction, threshold=0.5)
        print(f"✅ Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
        print("📸 Saving debug plots...")
        #save_debug_outputs(data, prediction, mask, out_dir=output_path, label="get_prediction")
        print("🛑 moving to predict_all:")

        if self._config2 is not None:
            print("📌 Secondary model is available. Running secondary_prediction...")
            print(f"↪ Before swap_to_original_axis (mask): shape={mask.shape}, dtype={mask.dtype}, unique={np.unique(mask)}")
            swapped_mask = swap_to_original_axis(swap_axis, mask)
            print(f"✅ Swapped mask shape: {swapped_mask.shape}, dtype: {swapped_mask.dtype}, unique: {np.unique(swapped_mask)}")
            prediction_all_path=os.path.join(output_path, 'prediction_all.nii.gz')
            print(f"💾 Saving swapped mask to: {prediction_all_path}")
            save_nifti(np.int16(swapped_mask), prediction_all_path)
           
            print("🧠 Running secondary prediction...")
            prediction = secondary_prediction(mask, vol=nifti_data.astype(float),
                                              config2=self._config2, model2=self._model2,
                                              preprocess_method2="window_1_99", norm_params2=self._norm_params2,
                                              overlap_factor=overlap_factor, augment2=None,
                                              num_augment=1,
                                              return_all_preds=None)
            print(f"✅ Secondary prediction done. Shape: {prediction.shape}, dtype: {prediction.dtype}, min/max: {prediction.min()}/{prediction.max()}")
            prediction_binarized = postprocess_prediction(prediction, threshold=0.5)
            print(f"✅ Postprocessed secondary prediction (binary). Shape: {prediction_binarized.shape}, unique: {np.unique(prediction_binarized)}")
            prediction_binarized = swap_to_original_axis(swap_axis, prediction_binarized)
            print(f"↩ Swapped back final binary prediction. Shape: {prediction_binarized.shape}, unique: {np.unique(prediction_binarized)}")
            # 🧪 Save debug plots for secondary prediction
            # save_debug_outputs(
            #     data=nifti_data,  # original input before axis move
            #     prediction=prediction,
            #     mask=prediction_binarized,
            #     out_dir=output_path,
            #     label="secondary_prediction"
            # )
            final_pred_path = os.path.join(output_path, 'prediction.nii.gz')
            print(f"💾 Saving final prediction to: {final_pred_path}")
            save_nifti_pred(np.int16(prediction_binarized), final_pred_path, reference_img_path=in_file)
          
        else:  # if there is no secondary prediction, save the first network prediction or predictions as the final ones
            print("⚠️ No secondary model available. Saving first-stage mask directly.")
            print(f"↪ Before swap_to_original_axis (mask): shape={mask.shape}, dtype={mask.dtype}, unique={np.unique(mask)}")
            mask = swap_to_original_axis(swap_axis, mask)
            print(f"✅ Swapped mask shape: {mask.shape}, dtype: {mask.dtype}, unique: {np.unique(mask)}")
            final_pred_path = os.path.join(output_path, 'prediction.nii.gz')
            print(f"💾 Saving primary prediction to: {final_pred_path}")
            save_nifti_pred(np.int16(mask), final_pred_path, reference_img_path=in_file)
           
        print('📝 Saving complete to {}'.format(output_path))
        print('✅ Prediction process finished.')
# optional funcion for saving outputs in the way for debuggin
# def save_debug_outputs(data, prediction, mask, out_dir="debug", label=""):
#     os.makedirs(out_dir, exist_ok=True)

#     suffix = f"_{label}" if label else ""

#     # # Save middle slice of input volume
#     # mid_input = data[:, :, data.shape[2] // 2]
#     # plt.figure(figsize=(8, 6))
#     # plt.imshow(mid_input, cmap='gray')
#     # plt.title(f"Middle Slice of Input Volume{suffix}")
#     # input_path = os.path.join(out_dir, f"input_middle_slice{suffix}.png")
#     # plt.savefig(input_path)
#     # plt.close()

#     # Save middle slice of raw prediction
#     mid_pred = prediction[:, :, prediction.shape[2] // 2]
#     plt.figure(figsize=(8, 6))
#     plt.imshow(mid_pred, cmap='gray')
#     plt.title(f"Middle Slice of Raw Prediction{suffix}")
#     pred_path = os.path.join(out_dir, f"prediction_middle_slice{suffix}.png")
#     plt.savefig(pred_path)
#     plt.close()

   

#     # Save middle slice of binary mask
#     mid_mask = mask[:, :, mask.shape[2] // 2]
#     plt.figure(figsize=(8, 6))
#     plt.imshow(mid_mask, cmap='gray')
#     plt.title(f"Middle Slice of Binary Mask{suffix}")
#     mask_path = os.path.join(out_dir, f"mask_middle_slice{suffix}.png")
#     plt.savefig(mask_path)
#     plt.close()

#     return pred_path, mask_path
