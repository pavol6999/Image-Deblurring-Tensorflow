import tensorflow as tf
from keras import backend as K


def PSNR(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, 1.0))


def SSIM(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
