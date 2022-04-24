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

        activation2 = tf.keras.layers.Activation(activation_func)(conv2)

        # residual wizardry
        shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding)(input_layer)
        shortcut = tf.keras.layers.BatchNormalization(axis=3)(shortcut)
        out_layer = tf.keras.layers.add([shortcut, activation2])

    return out_layer


def AttentionBlock(input_layer, gating, shape):
    x_theta = tf.keras.layers.Conv2D(shape, (1, 1), padding="same")(input_layer)
    g_phi = tf.keras.layers.Conv2D(shape, (1, 1), padding="same")(gating)

    addition_xg = tf.keras.layers.add([g_phi, x_theta])

    activation_xg = tf.keras.layers.Activation("relu")(addition_xg)

    psi = tf.keras.layers.Conv2D(1, (1, 1), padding="same")(activation_xg)

    sigmoid_psi = tf.keras.layers.Activation("sigmoid")(psi)

    output = tf.keras.layers.multiply([input_layer, sigmoid_psi])

    return output


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

        ### HERE LIES THE ATTENTION BLOCK ###
        inter_shape = tf.keras.backend.int_shape(input_layer)[3]
        att_block = AttentionBlock(concat_layer, dropout, shape=inter_shape // 4)

        #####################################

        concat = tf.keras.layers.concatenate([att_block, concat_layer])

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

        shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding)(concat)
        shortcut = tf.keras.layers.BatchNormalization(axis=3)(shortcut)

        out_layer = tf.keras.layers.add([shortcut, activation3])

    return out_layer
