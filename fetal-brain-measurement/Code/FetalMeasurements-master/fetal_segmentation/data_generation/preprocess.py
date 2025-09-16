import SimpleITK as sitk
import numpy as np
from scipy import ndimage


def window_1_99(data, min_percent=1, max_percent=99):
    image = sitk.GetImageFromArray(data)
    image = sitk.IntensityWindowing(image,
                                    np.percentile(data, min_percent),
                                    np.percentile(data, max_percent))
    return sitk.GetArrayFromImage(image)


def normalize_data(data, mean, std):
    data -= mean
    data /= std
    return data


def norm_minmax(d):
    return -1 + 2 * (d - d.min()) / (d.max() - d.min())


def laplace(d):
    return ndimage.laplace(d)


def laplace_norm(d):
    return norm_minmax(laplace(d))


def grad(d):
    return ndimage.gaussian_gradient_magnitude(d, sigma=(1,1,1))


def grad_norm(d):
    return norm_minmax(grad(d))