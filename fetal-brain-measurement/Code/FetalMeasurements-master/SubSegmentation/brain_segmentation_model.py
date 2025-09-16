# --- Must go first ---
import torch
import functools
from torch.optim import Adam
from SubSegmentation.lovasz import lovasz_softmax  # Make sure lovasz_softmax is explicitly imported
from .processing_utils import pre_processing, pre_processing_no_tta, post_processing, majority_vote, _contrast
from nibabel import load as load_nii
import numpy as np
# Define your custom functions before loading model
def acc_no_bg(input, target):
    target = target.squeeze(1)
    mask = target != 0
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()

# ----------------------

# Now it's safe to import fastai and your custom module
import sys
sys.path.append('/workspace/fetal-brain-measurement/Code/FetalMeasurements-master/SubSegmentation')

from lovasz import lovasz_softmax
from processing_utils import acc_no_bg

from fastai.basic_train import load_learner

import os
from .lovasz import *  # Ensure these imports are valid
from .processing_utils import *  # Same with other imports
import __main__


class BrainSegmentationModel(object):
    def __init__(self, input_size, model_path, model_name, device='cuda'):
        self.image_size = input_size  # ✅ Add this line
        if not model_name.endswith('.pkl'):
            model_name += '.pkl'
        print("model path + model name:",model_path, model_name)
        learner_path = os.path.join(model_path, model_name)
        #print(f"[DEBUG] Trying to load model from: {learner_path}")
        try:
            learn = load_learner(model_path, file=model_name)
            if torch.cuda.is_available():
                learn.model.to(torch.device('cuda'))
                #print("[DEBUG] Model loaded on GPU.")
            else:
                learn.model.to(torch.device('cpu'))
                #print("[DEBUG] Model loaded on CPU.")
            self.model = learn
        except Exception as e:
            print(f"[ERROR] Failed to load learner from {learner_path}")
            raise e

    def predict_nifti(self, nifti_fdata, dest_filename, tta=True):
        # — normalize intensities by 1st–99th percentile window
        low, high = np.percentile(nifti_fdata, (1, 99))
        nifti_fdata = _contrast(nifti_fdata, low, high)
        if tta:
            self._predict_with_tta(nifti_fdata, dest_filename)
        else:
            self._predict_no_tta(nifti_fdata, dest_filename)

    def _predict_with_tta(self, nifti_fdata, dest_filename):
        #print("pre pre processing")
        images, min_ax, zeros, x_ax, y_ax = pre_processing(nifti_fdata, self.image_size)
        #print("🧪 DEBUG: Checking TTA input...")
        for i, aug in enumerate(images):
            stats = torch.cat([t.data.view(-1) for t in aug])
            #print(f"   ➤ Aug {i}: min={stats.min():.3f}, max={stats.max():.3f}, mean={stats.mean():.3f}")
        #print("post pre processing")
        rotations_results = []
        for rotated_images in images:
            rotated_result = []
            for image in rotated_images:
                pred_class, pred_idx, outputs = self.model.predict(image)
                #print("   🔹 outputs shape:", outputs.shape)
                #print("   🔹 outputs min/max:", outputs.min(), outputs.max())
                #print("🔍 pred_idx unique values:", np.unique(pred_idx.data.numpy()))
                rotated_result.append(pred_idx.data.squeeze())
                #if len(rotated_result) == 1:
                    #print("Intermediate prediction unique values:", np.unique(rotated_result[0]))
            rotations_results.append(rotated_result)
        #print("after for loop in with tta ")
        # Count label distribution per rotated result
        for i, rot in enumerate(rotations_results):
            flat = np.concatenate([r.flatten() for r in rot])
            classes, counts = np.unique(flat, return_counts=True)
            #print(f"   🧮 Rotation {i}: label counts = {dict(zip(classes, counts))}")

        segmentations_result = majority_vote(rotations_results)
        print("after majority vote in with tta ")
        post_processing(segmentations_result, min_ax, zeros, x_ax, y_ax, dest_filename, self.image_size)
        print("post post processing")
        
        # ✅ Add this block to debug:
        seg = load_nii(dest_filename).get_fdata()
        print("✅ DEBUG: Unique labels in saved segmentation:", np.unique(seg))

    def _predict_no_tta(self, nifti_fdata, dest_filename):
        images, min_ax, zeros, x_ax, y_ax = pre_processing_no_tta(nifti_fdata, self.image_size)
        segmentations_result = []

        for image in images:
            pred_class, pred_idx, outputs = self.model.predict(image)
            segmentations_result.append(pred_idx.data.squeeze())

        post_processing(segmentations_result, min_ax, zeros, x_ax, y_ax, dest_filename, self.image_size)
