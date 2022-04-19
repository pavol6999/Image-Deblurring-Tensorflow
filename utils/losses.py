import tensorflow as tf

sobelFilter = tf.keras.backend.variable(
    [
        [[[1.0, 1.0]], [[0.0, 2.0]], [[-1.0, 1.0]]],
        [[[2.0, 0.0]], [[0.0, 0.0]], [[-2.0, 0.0]]],
        [[[1.0, -1.0]], [[0.0, -2.0]], [[-1.0, -1.0]]],
    ]
)


def expand_sobel_operator(tensor):

    inputChannels = tf.keras.backend.reshape(
        tf.keras.backend.ones_like(tensor[0, 0, 0, :]), (1, 1, -1, 1)
    )
    return sobelFilter * inputChannels


def SobelLoss(y_true, y_pred):

    filt = expand_sobel_operator(y_true)

    sobel_true = tf.keras.backend.depthwise_conv2d(y_true, filt)
    sobel_pred = tf.keras.backend.depthwise_conv2d(y_pred, filt)

    return tf.keras.backend.mean(tf.keras.backend.square(sobel_true - sobel_pred))


def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


# taken from https://gist.github.com/quocdat32461997/cae85b748ce651ff6e3013880a5659af
def MeanGradientError(outputs, targets, weight=1.0):
    filter_x = tf.tile(
        tf.expand_dims(
            tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=outputs.dtype), axis=-1
        ),
        [1, 1, outputs.shape[-1]],
    )
    filter_x = tf.tile(tf.expand_dims(filter_x, axis=-1), [1, 1, 1, outputs.shape[-1]])
    filter_y = tf.tile(
        tf.expand_dims(
            tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=outputs.dtype), axis=-1
        ),
        [1, 1, targets.shape[-1]],
    )
    filter_y = tf.tile(tf.expand_dims(filter_y, axis=-1), [1, 1, 1, targets.shape[-1]])

    # output gradient
    output_gradient_x = tf.math.square(tf.nn.conv2d(outputs, filter_x, strides=1, padding="SAME"))
    output_gradient_y = tf.math.square(tf.nn.conv2d(outputs, filter_y, strides=1, padding="SAME"))

    # target gradient
    target_gradient_x = tf.math.square(tf.nn.conv2d(targets, filter_x, strides=1, padding="SAME"))
    target_gradient_y = tf.math.square(tf.nn.conv2d(targets, filter_y, strides=1, padding="SAME"))

    # square
    output_gradients = tf.math.sqrt(tf.math.add(output_gradient_x, output_gradient_y))
    target_gradients = tf.math.sqrt(tf.math.add(target_gradient_x, target_gradient_y))

    # compute mean gradient error
    shape = output_gradients.shape[1:3]
    mge = tf.math.reduce_sum(
        tf.math.squared_difference(output_gradients, target_gradients) / (shape[0] * shape[1])
    )

    return mge * weight
