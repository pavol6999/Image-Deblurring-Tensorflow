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

        # if not os.path.exists(self.filename):
        #     self.model.stop_training = True
        #     raise (ValueError("The filename {} does not exist".format(self.filename)))

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
    Callback which will be called after each epoch of training to predict iamges from a directory (default `predict` dir).
    """

    def __init__(self, dir, wandb_enabled):
        super().__init__()
        self.wandb_enabled = wandb_enabled
        self.dir = dir
        self.predict_imgs = []
        for file in PredictImageAfterEpoch._files(dir):
            img = tf.keras.preprocessing.image.load_img(dir + "/" + file)
            img = np.array(img) / 255.0
            self.predict_imgs.append(img)

        if self.wandb_enabled:

            wandb.log({"visualization": [wandb.Image(img) for img in self.predict_imgs]})
        else:
            if not os.path.exists(f"{dir}/predicted"):
                os.makedirs(f"{dir}/predicted")

    @staticmethod
    def _files(path):
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                yield file

    def write_image(self, images, epoch):

        if self.wandb_enabled:
            wandb.log({"visualization": [wandb.Image(img) for img in images]})
        else:
            for i, img in enumerate(images):
                plt.imsave(f"{self.dir}/predicted/epoch_{epoch}_{i}.png", img[0])

    def on_epoch_end(self, epoch, logs={}):
        # D:\FIIT\BP\training_set\train\blur\data

        predicted_images = []
        for img in self.predict_imgs:
            predicted = np.expand_dims(img, 0)
            predicted = self.model(predicted, training=False)
            predicted = np.copy(predicted)
            predicted_images.append(predicted)
        self.write_image(predicted_images, epoch)
