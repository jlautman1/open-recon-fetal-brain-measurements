from scipy.special import entr
from scipy.ndimage import label, labeled_comprehension
import numpy as np


def entropy(d, thres=0.5, axis=0):
    n_samples = d.shape[axis]
    d = d > thres
    d_sum = np.sum(d, axis=axis) / n_samples
    ent_0 = entr(1.0-d_sum)
    ent_1 = entr(d_sum)
    return ent_0 + ent_1


# def entropy2(d, axis=0):
#     prob_1 = np.mean(d, axis=0)
#     ent_1 = prob_1 * np.log(prob_1)
#     prob_0 = np.mean(1-d, axis=0)
#     ent_0 = prob_0 * np.log(prob_0)
#     return -(np.nan_to_num(ent_0) + np.nan_to_num(ent_1))