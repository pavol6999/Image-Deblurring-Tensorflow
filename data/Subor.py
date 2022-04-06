# blurred_img_paths, ground_truth_img_paths = [], []
# # nacitat cesty k obrazkom
# blurred_images, ground_truth_images = [], []
# #nacitat obrazky do arrays

# def train_generator():
#     for i in range(len(blurred_img_paths)):
#         yield blurred_images[i], ground_truth_images[i]


# def dataset_from_generator(generator):
#     return tf.data.Dataset.from_generator(
#         generator,
#         output_types=(tf.float32, tf.float32),
#         output_shapes=((None, None, 3), (None, None, 3))
#     )

# train_dataset = dataset_from_generator(train_generator)


import cv2, os
import keras
import tensorflow as tf
from keras import layers


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    input_layer = keras.Input(shape=(None, None, 3))

    out = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(input_layer)

    conv_model = keras.Model(input_layer, out)
    conv_model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

conv_model.summary()

path = "D:/FIIT/BP/training_set/GOPR0372_07_00"

blurred_imgs, ground_truth_imgs = [], []

blurred_imgs = [
    cv2.imread(os.path.join(path, "blur", f)) / 255 for f in os.listdir(os.path.join(path, "blur"))
]
ground_truth_imgs = [
    cv2.imread(os.path.join(path, "sharp", f)) / 255
    for f in os.listdir(os.path.join(path, "sharp"))
]


def data_generator():
    for i in range(len(blurred_imgs)):
        yield blurred_imgs[i], ground_truth_imgs[i]


dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=((None, None, 3), (None, None, 3)),
).batch(1)

conv_model.fit(dataset, epochs=2, validation_data=dataset)
print("Done")
