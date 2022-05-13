# This file contains the custom callback/s for the training of the model
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from datetime import datetime
import wandb


class SaveTrainedModel(tf.keras.callbacks.Callback):
    """Callback to save the model after exiting training.

    `SaveTrainedModel` callback is used in conjunction with training using
    `model.fit()` to save a model at the end of training, so the model can be loaded later to predict or evaluate.
    from the state saved.


        Args:
            filename: string, name to save the model file. e.g.
                filename = os.path.join(working_dir, 'models', file_name).
            date: int, 0 or 1. Whether to save the model with the current date appended as a prefix.
    """

    def __init__(self, filename, date):
        super().__init__()
        self.filename = filename
        self.date = date

        if date > 0:
            self.fullname = (
                str(datetime.now().strftime("%Y-%m-%d-%H%M")) + "_" + self.filename + ".h5"
            )

        else:
            self.fullname = self.filename + ".h5"

    def on_train_begin(self, logs=None):
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] Starting training; Model will be saved as {self.fullname}"
        )

    def on_train_end(self, logs=None):
        self.model.save(os.getcwd() + "/models/trained_models/" + self.fullname)
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] Stop training; Model saved as {self.fullname}"
        )


import numpy as np


from tensorflow.keras.callbacks import Callback


class PredictImageAfterEpoch(Callback):
    """
    Callback object which will be called after each epoch in training phase to predict images from a directory (default `predict` dir).
    """

    def __init__(self, dir, wandb_enabled):
        super().__init__()
        self.wandb_enabled = wandb_enabled
        self.dir = dir
        self.predict_imgs = []

        # load the images, that we want to predict after each epoch
        for file in PredictImageAfterEpoch._files(dir):
            img = tf.keras.preprocessing.image.load_img(dir + "/" + file)
            img = np.array(img) / 255.0  # convert to numpy array and normalize
            self.predict_imgs.append(img)

        # if we log our data to WandB, then dont save the image but upload it to the WandB project
        if self.wandb_enabled:

            wandb.log({"visualization": [wandb.Image(img) for img in self.predict_imgs]})

        # else prepare a directory, where the ouputted predicted images will be saved
        else:
            if not os.path.exists(f"{dir}/predicted"):
                os.makedirs(f"{dir}/predicted")

    @staticmethod
    def _files(path):
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                yield file

    def write_image(self, images, epoch):
        """Writes the image data to the specified directory, with name as `epoch_(epoch_number)_(i).png`.
        Args:
            images: array of images to write
            epoch: current epoch
        """

        # if we log our data to WandB, then dont save the image but upload it to the WandB project
        if self.wandb_enabled:
            wandb.log({"visualization": [wandb.Image(img) for img in images]})

        # else write image data
        else:
            for i, img in enumerate(images):
                plt.imsave(f"{self.dir}/predicted/epoch_{epoch}_{i}.png", img[0])

    def on_epoch_end(self, epoch, logs={}):

        if epoch % 5 == 0:
            predicted_images = []
            for img in self.predict_imgs:
                predicted = np.expand_dims(img, 0)  # add batch information
                predicted = self.model(predicted, training=False)  # predict the image
                predicted = np.copy(predicted)  # transform it to a numpy array
                predicted_images.append(
                    predicted
                )  # append it to an array with predicted image data
            self.write_image(predicted_images, epoch)
