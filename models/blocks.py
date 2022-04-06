import tensorflow as tf


def ConvBlock(
    input_layer,
    activation_func,
    filters=32,
    kernel_size=(3, 3),
    kernel_init="glorot_uniform",
    bias_init="zeros",
    padding="valid",
    dropout_chance=0.1,
    block_name="ConvBlock",
):
    with tf.name_scope(block_name):
        conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            padding=padding,
        )(input_layer)
        bn = tf.keras.layers.BatchNormalization()(conv)
        activation = tf.keras.layers.Activation(activation_func)(bn)

        dropout = tf.keras.layers.Dropout(dropout_chance)(activation)

        conv2 = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            padding=padding,
        )(dropout)
        bn2 = tf.keras.layers.BatchNormalization()(conv2)
        unet_layer_out = tf.keras.layers.Activation(activation_func)(bn2)

        max_pooling = tf.keras.layers.MaxPooling2D((2, 2))(unet_layer_out)

    return max_pooling, unet_layer_out


def ConvBlockTranspose(
    input_layer,
    concat_layer,
    activation_func,
    filters=32,
    kernel_size=(3, 3),
    kernel_init="glorot_uniform",
    bias_init="zeros",
    padding="valid",
    dropout_chance=0.1,
    block_name="ConvBlockTranspose",
):
    with tf.name_scope(block_name):

        conv_transpose = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=(2, 2),
            padding=padding,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )(input_layer)
        bn = tf.keras.layers.BatchNormalization()(conv_transpose)
        activation = tf.keras.layers.Activation(activation_func)(bn)

        dropout = tf.keras.layers.Dropout(dropout_chance)(activation)

        concat = tf.keras.layers.concatenate([dropout, concat_layer])

        conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            padding=padding,
        )(concat)
        bn2 = tf.keras.layers.BatchNormalization()(conv)
        activation2 = tf.keras.layers.Activation(activation_func)(bn2)

        conv2 = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            padding=padding,
        )(activation2)
        bn3 = tf.keras.layers.BatchNormalization()(conv2)
        out_layer = tf.keras.layers.Activation(activation_func)(bn3)

    return out_layer
