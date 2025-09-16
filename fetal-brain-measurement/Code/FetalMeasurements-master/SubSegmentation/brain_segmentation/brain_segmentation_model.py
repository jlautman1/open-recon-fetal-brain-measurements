from .lovasz import *
from .processing_utils import *
import sys
sys.path.append('/workspace/fetal-brain-measurement/Code/FetalMeasurements-master/SubSegmentation')

from lovasz import lovasz_softmax
from processing_utils import acc_no_bg

from fastai.basic_train import load_learner

learn = load_learner(model_path, model_name)


class BrainSegmentationModel(object):
    def __init__(self, image_size, model_path, model_name):
        self.model = load_learner(model_path, model_name)
        self.image_size = image_size

    def predict_nifti(self, nifti_fdata, dest_filename, tta=True):
        if tta:
            self._predict_with_tta(nifti_fdata, dest_filename)
        else:
            self._predict_no_tta(nifti_fdata, dest_filename)

    def _predict_with_tta(self, nifti_fdata, dest_filename):
        images, min_ax, zeros, x_ax, y_ax = pre_processing(nifti_fdata, self.image_size)
        rotations_results = []
        for rotated_images in images:
            rotated_result = []
            for image in rotated_images:
                pred_class, pred_idx, outputs = self.model.predict(image)
                rotated_result.append(pred_idx.data.squeeze())
            rotations_results.append(rotated_result)

        segmentations_result = majority_vote(rotations_results)
        post_processing(segmentations_result, min_ax, zeros, x_ax, y_ax, dest_filename, self.image_size)

    def _predict_no_tta(self, nifti_fdata, dest_filename):
        images, min_ax, zeros, x_ax, y_ax = pre_processing_no_tta(nifti_fdata, self.image_size)
        segmentations_result = []

        for image in images:
            pred_class, pred_idx, outputs = self.model.predict(image)
            segmentations_result.append(pred_idx.data.squeeze())

        post_processing(segmentations_result, min_ax, zeros, x_ax, y_ax, dest_filename, self.image_size)


