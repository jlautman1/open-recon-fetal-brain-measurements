from scipy.ndimage.measurements import label
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes
import numpy as np


def get_main_connected_component(data):
#    s = np.ones((3, 3, 3), dtype=int)
    labeled_array, num_features = label(data)
    i = np.argmax([np.sum(labeled_array == _) for _ in range(1, num_features + 1)]) + 1
    return labeled_array == i


def postprocess_prediction(pred, threshold=0.5, fill_holes=False, connected_component=True):

    pred = pred > threshold

    if(np.count_nonzero(pred)==0):#no nonzero elements
        return pred

    if fill_holes:#not used by default
        pred = binary_fill_holes(pred)

    if connected_component:
        pred = get_main_connected_component(pred)
    return pred
