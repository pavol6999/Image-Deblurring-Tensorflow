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
    Callback which will be called after each epoch of training to predict a patch from test dataset generator.
    At the start of training, the test dataset will yield one image pair patches. Then after each epoch, the blurred image patch will be predicted and saved.
    """

    def __init__(self, image_pair, wandb_enabled):
        super().__init__()
        self.wandb_enabled = wandb_enabled
        self.blur, self.sharp = next(iter(image_pair))
        tf.keras.utils.save_img(
            os.getcwd() + f"/images/{datetime.now().strftime('%Y-%m-%d')}_blur_visualization.jpg",
            self.blur[0],
            scale=True,
        )

        tf.keras.utils.save_img(
            os.getcwd() + f"/images/{datetime.now().strftime('%Y-%m-%d')}_sharp_visualization.jpg",
            self.sharp[0],
            scale=True,
        )
        if self.wandb_enabled:
            image_blur = wandb.Image(self.blur[0])
            image_sharp = wandb.Image(self.sharp[0])
            wandb.log({"examples": [image_blur, image_sharp]})

    def write_image(self, image, epoch):
        image_to_write = np.copy(image)
        tf.keras.utils.save_img(
            os.getcwd() + f"/images/{datetime.now().strftime('%Y-%m-%d')}_predicted_{epoch}.jpg",
            image_to_write[0],
            scale=True,
        )
        if self.wandb_enabled:
            img = wandb.Image(
                image_to_write[0],
                caption="f{datetime.now().strftime('%Y-%m-%d')}_predicted_epoch_{epoch}",
            )
            wandb.log({"predicted": img})

    def on_epoch_end(self, epoch, logs={}):
        deblurred_image = self.model(self.blur, training=False)
        self.write_image(deblurred_image, epoch)
