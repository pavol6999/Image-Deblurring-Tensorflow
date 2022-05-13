import tensorflow as tf


def SSIM_L1_Loss(y_true, y_pred):
    """Custom loss function that incorporates L1 error function (MSE) and the inverse of the SSIM metric. Therefore this loss function also includes an alpha
    constant that divides the partake of these two loss functions."""
    ALPHA = 0.84
    return ALPHA * SSIMLoss(y_true, y_pred) + (1 - ALPHA) * tf.keras.metrics.mean_absolute_error(
        y_true, y_pred
    )


def SSIMLoss(y_true, y_pred):
    """Plain SSIM loss function, which uses the inverse of the SSIM metric."""
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
