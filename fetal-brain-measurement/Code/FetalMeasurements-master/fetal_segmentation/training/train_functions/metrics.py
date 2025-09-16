from functools import partial
from keras.losses import binary_crossentropy
from keras import backend as K
import tensorflow as tf


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_distance_weighted(y_true, y_pred, distance_map, smooth=1., beta=0.2):
    """
    Weight dice by distance map in order to give more weight to voxels in the contour
    :param y_true:
    :param y_pred:
    :param distance_map: Distance map from graound truth contour. Minimum value is 1
    :param smooth:
    :return:
    """
    #compute distance weighted dice
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    distance_map = K.flatten(distance_map)
    y_true_f = y_true_f/(K.log(distance_map) + smooth)
    y_pred_f = y_pred_f/(K.log(distance_map) + smooth)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_distance_weighted = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    dice = dice_coefficient(y_true, y_pred)

    return (beta * dice_distance_weighted) + ((1-beta) * dice)


def double_dice_loss(y_true, y_pred, ratio=10.0):
    return -dice_coefficient(y_true, y_pred) + ratio*dice_coefficient(1-y_true, y_pred)


def vod_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def dice_distance_weighted_loss(distance_map):
    def loss(y_true, y_pred):
        return -dice_distance_weighted(y_true, y_pred, distance_map)
    return loss


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def vod_coefficient_loss(y_true, y_pred):
    return -vod_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth / 2) / (K.sum(y_true,
                                                                axis=axis) + K.sum(y_pred,
                                                                                   axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[..., label_index], y_pred[..., label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


def dice_and_xent(y_true, y_pred, xent_weight=1.0, weight_mask=None):
    return dice_coef_loss(y_true, y_pred) + \
           xent_weight * weighted_cross_entropy_loss(y_true, y_pred, weight_mask)


def weighted_cross_entropy_loss(y_true, y_pred, weight_mask=None):
    xent = K.binary_crossentropy(y_true, y_pred)
    if weight_mask is not None:
        xent = K.prod(weight_mask, xent)
    return K.mean(xent)


def _focal_loss(gamma=2., alpha=.5):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
binary_crossentropy_loss = binary_crossentropy
focal_loss = _focal_loss()