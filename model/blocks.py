import tensorflow as tf


def ConvBlock(
    input_layer,
    activation_func,
    filters=32,
    kernel_size=(3, 3),
    kernel_init="he_normal",
    bias_init="zeros",
    padding="valid",
    block_name="ConvBlock",
):
    """Default Residual Convolutional block used in encoding steps witch batch normalization.

    Args:
        input_layer: Input to the convolutional block
        activation_func: activation function
        filters (int, optional): Defaults to 32.
        kernel_size (tuple, optional): Defaults to (3, 3).
        kernel_init (str, optional): Defaults to "he_normal".
        bias_init (str, optional): Defaults to "zeros".
        padding (str, optional): Defaults to "valid".
        block_name (str, optional): Defaults to "ConvBlock".

    Returns:
        A residual convolutional block
    """
    with tf.name_scope(block_name):

        # first small conv block
        conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            padding=padding,
        )(input_layer)
        bn = tf.keras.layers.BatchNormalization(axis=3)(conv)
        activation = tf.keras.layers.Activation(activation_func)(bn)

        # second small conv block
        conv2 = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            padding=padding,
        )(activation)

        activation2 = tf.keras.layers.Activation(activation_func)(conv2)

        # residual part of the convolutional block
        shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding)(input_layer)
        shortcut = tf.keras.layers.BatchNormalization(axis=3)(shortcut)
        out_layer = tf.keras.layers.add([shortcut, activation2])

    return out_layer


# attention convolutional block. Inspired by the attention block in the paper and https://www.youtube.com/watch?v=KOF38xAvo8I&t=546s.
def AttentionBlock(input_layer, gating, shape):

    # signal coming from skip connection
    x_theta = tf.keras.layers.Conv2D(shape, (1, 1), padding="same")(input_layer)

    # gating signal from the decoding block
    g_phi = tf.keras.layers.Conv2D(shape, (1, 1), padding="same")(gating)

    # add the x signal and the gating signal
    addition_xg = tf.keras.layers.add([g_phi, x_theta])

    # apply the relu activation function
    activation_xg = tf.keras.layers.Activation("relu")(addition_xg)

    # apply the 1x1 convolution with same padding to get a vector of crop_shape x crop_shape x 1

    # weights of the attention, they can be seen as the attention map with values between 0 and INF
    psi = tf.keras.layers.Conv2D(1, (1, 1), padding="same")(activation_xg)

    # bring the values to 0 to 1
    # 0 - not important, 1 - important
    sigmoid_psi = tf.keras.layers.Activation("sigmoid")(psi)

    # multiply the attention map with the input layer
    output = tf.keras.layers.multiply([input_layer, sigmoid_psi])

    return output


def ConvBlockTranspose(
    input_layer,
    concat_layer,
    activation_func,
    filters=32,
    kernel_size=(3, 3),
    kernel_init="he_normal",
    bias_init="zeros",
    padding="valid",
    block_name="ConvBlockTranspose",
):
    """Decoding residual convolutional block with batch normalization and attention layer.

    Args:
        input_layer: Input layer to the convolutional decoding block
        concat_layer: _description_
        activation_func: _description_
        filters (int, optional): Defaults to 32.
        kernel_size (tuple, optional): Defaults to (3, 3).
        kernel_init (str, optional): Defaults to "he_normal".
        bias_init (str, optional): Defaults to "zeros".
        padding (str, optional): Defaults to "valid".
        block_name (str, optional): Defaults to "ConvBlockTranspose".

    Returns:
        _type_: _description_
    """
    with tf.name_scope(block_name):

        conv_transpose = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=(2, 2),
            padding=padding,
            activation="relu",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )(input_layer)
        bn = tf.keras.layers.BatchNormalization(axis=3)(conv_transpose)
        activation = tf.keras.layers.Activation(activation_func)(bn)

        # dropout = tf.keras.layers.Dropout(dropout_chance)(activation)

        ### HERE LIES THE ATTENTION BLOCK ###
        inter_shape = tf.keras.backend.int_shape(input_layer)[3]
        att_block = AttentionBlock(concat_layer, activation, shape=inter_shape // 4)

        #####################################

        concat = tf.keras.layers.concatenate([att_block, concat_layer], axis=3)

        conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            padding=padding,
        )(concat)
        bn2 = tf.keras.layers.BatchNormalization(axis=3)(conv)
        activation2 = tf.keras.layers.Activation(activation_func)(bn2)

        conv2 = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            padding=padding,
        )(activation2)

        activation3 = tf.keras.layers.Activation(activation_func)(conv2)

        # return part of the blcok
        shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding)(concat)
        shortcut = tf.keras.layers.BatchNormalization(axis=3)(shortcut)

        out_layer = tf.keras.layers.add([shortcut, activation3])

    return out_layer
