import tensorflow as tf


def ConvBlock(
    input_layer,
    activation_func,
    filters=32,
    kernel_size=(3, 3),
    kernel_init="glorot_uniform",
    bias_init="zeros",
    padding="valid",
    dropout_chance=0.3,
    block_name="ConvBlock",
):
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

        # dropout
        dropout = tf.keras.layers.Dropout(dropout_chance)(activation)

        # second small conv block
        conv2 = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            padding=padding,
        )(dropout)
        bn2 = tf.keras.layers.BatchNormalization(axis=3)(conv2)
        activation2 = tf.keras.layers.Activation(activation_func)(bn2)

        # residual wizardry
        shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding)(input_layer)
        shortcut = tf.keras.layers.BatchNormalization(axis=3)(shortcut)
        out_layer = tf.keras.layers.add([shortcut, activation2])

    return out_layer


def ConvBlockTranspose(
    input_layer,
    concat_layer,
    activation_func,
    filters=32,
    kernel_size=(3, 3),
    kernel_init="glorot_uniform",
    bias_init="zeros",
    padding="valid",
    dropout_chance=0.3,
    block_name="ConvBlockTranspose",
):
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

        dropout = tf.keras.layers.Dropout(dropout_chance)(activation)

        concat = tf.keras.layers.concatenate([dropout, concat_layer])

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
        bn3 = tf.keras.layers.BatchNormalization(axis=3)(conv2)
        activation3 = tf.keras.layers.Activation(activation_func)(bn3)

        shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding)(concat)
        shortcut = tf.keras.layers.BatchNormalization(axis=3)(shortcut)
        shortcut = tf.keras.layers.Activation(activation_func)(shortcut)

        out_layer = tf.keras.layers.add([shortcut, activation3])

    return out_layer
