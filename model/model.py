from datetime import datetime
import math
from types import SimpleNamespace
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import wandb
from wandb.keras import WandbCallback
import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

sys.path.append("./")
from model.blocks import ConvBlock, ConvBlockTranspose
from data.DataGenerator import DataGenerator
from utils.callbacks import PredictImageAfterEpoch, SaveTrainedModel
from utils.metrics import PSNR, SSIM
from utils.losses import SSIM_L1_Loss, SSIMLoss


class DeblurModel:
    """
    Base class for deblur model architecture based on the U-Net architecture.
    """

    def __init__(self, args, config=None):
        self.train_phase = args.train

        ############## Default config ################
        self.learning_rate = 0.003
        self.optimizer = "adam"
        self.batch_size = 4
        self.kernel_size = 3
        self.loss_function = "mse"
        self.filters = 32
        ##############################################

        self.args = args or None
        self.epochs = args.epochs  # or 10
        self.batch_size = args.batch_size  # or 1

        self.data = args.data

        self.use_best_weights = args.continue_training
        self.early_stopping = args.early_stopping
        self.save_trained_model = args.save_after_train
        self.checkpoint = args.checkpoints
        if self.checkpoint:
            self.checkpoint_dir = args.checkpoint_dir
        self.epoch_visualization = args.epoch_visualization
        self.tensorboard = args.tensorboard
        self.wandb = args.wandb
        self.patience = args.patience

        self.test_phase = args.test

        if self.test_phase:
            self.load_model(args.model_path)

        ## Override default config with user config ##
        if config is not None:
            self.learning_rate = config["learning_rate"]  #
            self.optimizer = config["optimizer"]  #

            self.batch_size = config["batch_size"]  #
            self.kernel_size = config["kernel_size"]  #
            self.loss_function = config["loss_function"]
            self.filters = config["filters"]  #

        ##############################################
        if self.wandb and config is None:

            wandb.init(
                entity="xkrajkovic", project="bp_deblur", sync_tensorboard=False
            )  # this intialization must be rewrittten if you wish to use your own wandb project and entity
            wandb.config.learning_rate = self.learning_rate
            wandb.config.optimizer = self.optimizer
            wandb.config.batch_size = self.batch_size
            wandb.config.kernel_size = self.kernel_size
            wandb.config.loss_function = self.loss_function
            wandb.config.filters = self.filters

        # if we are in training phase of the model, we need to create the dataset image generators of blur and sharp images
        # these are the arguments passed down to the Datagenerator class
        if self.train_phase:

            self.train_args = SimpleNamespace(
                shuffle=True,
                seed=1,
                data_path=f"{self.data}/train",
                channels=3,  # rgb
                noise=True,
                flip=True,
                batch_size=self.batch_size,
                mode="train",
                repeat=1,
            )
            self.test_args = SimpleNamespace(
                shuffle=True,
                seed=1,
                flip=False,
                noise=False,
                data_path=f"{self.data}/test",
                channels=3,  # rgb
                batch_size=self.batch_size,
                mode="test",
                repeat=1,
            )

    @staticmethod
    def _files(path):
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                yield file

    def load_model(self, model_path):
        """
        Loads the model from the specified path. Needs to include the model name as well.
        So if for example our trained model is in `trained` directory, we need to specify the path as `trained/model_name.h5`.
        """
        # create a dictionary containing the custom objects in our model as these are not saved in the training phase. In our case
        # the custom objects are the loss functions and the metrics
        custom_objects = {
            "SSIMLoss": SSIMLoss,
            "SSIM_L1_Loss": SSIM_L1_Loss,
            "PSNR": PSNR,
            "SSIM": SSIM,
        }
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    def save_model(self, out_path, model_name):
        """
        Saves the model to the specified path.
        """
        self.model.save(out_path + model_name)

    def callbacks_builder(self):
        """Method that builds the callbacks for the model."""

        def model_checkpoint(path):
            """
            We use the `save_best_only` with the reference metric being
                the `val_ssim` metric. After each epoch, if the `val_ssim` metric is better than the previous one, the model weights are saved.
            Args:
                path (str): path where during the training, the model weights will be saved.

            Returns:
                `tf.keras.callbacks.ModelCheckpoint`: callback that saves the model weights after each epoch.
            """
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(os.getcwd(), path),
                save_weights_only=True,
                monitor="val_ssim",
                mode="max",
                verbose=20,
                save_best_only=True,
            )
            return model_checkpoint_callback

        def early_stopping():
            """Callback maker to run early stopping.
                Default patience is 5.
            Returns:
                tf.keras.callbacks.EarlyStopping: callback that stops the training if the validation loss does not improve after a certain number of epochs.
            """
            return tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=self.patience,
                verbose=1,
                mode="auto",
                baseline=None,
                restore_best_weights=False,
            )

        def wandb_callback():
            """
            WandB callback. This callback is used to log the training metrics to wandb. After each 5 epochs, predictions are made on the test set and logged to wandb. Model with the best "val_loss" metric is saved.

            Returns:
                 WandbCallback
            """
            return WandbCallback(
                log_weights=False,
                generator=self.test_img_generator,
                validation_steps=10,
                predictions=10,
                input_type="images",
                output_type="images",
                log_evaluation=True,
                log_evaluation_frequency=5,
            )

        callbacks = []

        # append our callback array and return it
        if self.early_stopping:
            callbacks.append(early_stopping())

        if self.checkpoint:
            callbacks.append(model_checkpoint(self.checkpoint_dir))

        if self.save_trained_model:
            callbacks.append(SaveTrainedModel("deblurmodel", 1))

        if self.epoch_visualization:

            callbacks.append(PredictImageAfterEpoch("predict", self.wandb))

        if self.wandb:
            callbacks.append(wandb_callback())
        return callbacks

    def build(self, patch_size, num_layers=3):
        """Build the model architecture and compile it."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Building model...")

        # use of mirrored_strategy as the model was trained on a computer with multiple GPU. by doing this we can utilize more than one GPU
        mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():

            # enable the autoshard policy which shards the dataset
            if self.train_phase:
                options = tf.data.Options()
                options.experimental_distribute.auto_shard_policy = (
                    tf.data.experimental.AutoShardPolicy.DATA
                )

                self.train_img_generator = DataGenerator(self.train_args)().with_options(options)
                self.test_img_generator = DataGenerator(self.test_args)().with_options(options)

            ### ARCHITECTURE OF THE RESIDUAL ATTENTION U-NET ###
            input = tf.keras.layers.Input((None, None, 3))

            # downsample
            layer1 = ConvBlock(
                input,
                "relu",
                self.filters,
                block_name="Conv_Level_1",
                kernel_size=(self.kernel_size, self.kernel_size),
                padding="same",
            )
            maxpooling_1 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=2, padding="same"
            )(layer1)

            layer2 = ConvBlock(
                maxpooling_1,
                "relu",
                self.filters * 2,
                block_name="Conv_Level_2",
                kernel_size=(self.kernel_size, self.kernel_size),
                padding="same",
            )
            maxpooling_2 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=2, padding="same"
            )(layer2)

            layer3 = ConvBlock(
                maxpooling_2,
                "relu",
                self.filters * 4,
                block_name="Conv_Level_3",
                kernel_size=(self.kernel_size, self.kernel_size),
                padding="same",
            )
            maxpooling_3 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=2, padding="same"
            )(layer3)

            layer4 = ConvBlock(
                maxpooling_3,
                "relu",
                self.filters * 8,
                block_name="Conv_Level_4",
                kernel_size=(self.kernel_size, self.kernel_size),
                padding="same",
            )
            maxpooling_4 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=2, padding="same"
            )(layer4)

            # hrdlo

            # Block 5
            conv_throttle = tf.keras.layers.Conv2D(
                self.filters * 16,
                (self.kernel_size, self.kernel_size),
                padding="same",
                name="throttle_conv1",
            )(maxpooling_4)
            conv_throttle2 = tf.keras.layers.Conv2D(
                self.filters * 16,
                (self.kernel_size, self.kernel_size),
                padding="same",
                name="throttle_conv2",
            )(conv_throttle)
            conv_throttle3 = tf.keras.layers.Conv2D(
                self.filters * 16,
                (self.kernel_size, self.kernel_size),
                padding="same",
                name="throttle_conv3",
            )(conv_throttle2)

            # upsample

            layer_up = ConvBlockTranspose(
                conv_throttle3,
                layer4,
                "relu",
                self.filters * 8,
                padding="same",
                kernel_size=(self.kernel_size, self.kernel_size),
                block_name="Conv_Up_Level_4",
            )

            layer_up1 = ConvBlockTranspose(
                layer_up,
                layer3,
                "relu",
                self.filters * 4,
                padding="same",
                kernel_size=(self.kernel_size, self.kernel_size),
                block_name="Conv_Up_Level_3",
            )
            layer_up2 = ConvBlockTranspose(
                layer_up1,
                layer2,
                "relu",
                self.filters * 2,
                padding="same",
                kernel_size=(self.kernel_size, self.kernel_size),
                block_name="Conv_Up_Level_2",
            )
            layer_up3 = ConvBlockTranspose(
                layer_up2,
                layer1,
                "relu",
                self.filters,
                padding="same",
                kernel_size=(self.kernel_size, self.kernel_size),
                block_name="Conv_Up_Level_1",
            )

            # # output
            output = tf.keras.layers.Conv2D(3, (1, 1), activation="sigmoid")(layer_up3)

            # create our model and assign it to the `self.model` attribute
            self.model = tf.keras.Model(input, output, name="Deblur")

            # choose the optimizer and loss function to be used, then compile the model
            if self.optimizer == "adam":
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self.learning_rate,
                )
            elif self.optimizer == "sgd":
                optimizer = tf.keras.optimizers.SGD(
                    learning_rate=self.learning_rate,
                )
            if self.loss_function == "SSIMLOSS":
                self.model.compile(
                    optimizer=optimizer,
                    loss=SSIMLoss,
                    metrics=["accuracy", PSNR, SSIM],
                )

            elif self.loss_function == "mse":
                self.model.compile(
                    optimizer=optimizer,
                    loss="mse",
                    metrics=["accuracy", PSNR, SSIM],
                )

            elif self.loss_function == "SSIM_L1_Loss":
                self.model.compile(
                    optimizer=optimizer,
                    loss=SSIM_L1_Loss,
                    metrics=["accuracy", PSNR, SSIM],
                )

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Finished Building")

        return self

    """
     Predicts the images in the directory 'predict' and saves the results in the directory 'predict/predicted'
    """

    def predict(self):
        predict_imgs = []  # images to be predicted
        predicted_images = []  # images predicted
        dir = "predict"

        # load the images inside the directory
        for file in DeblurModel._files(dir):
            img = tf.keras.preprocessing.image.load_img(dir + "/" + file)
            img = np.array(img) / 255.0  # max value of the color channel
            predict_imgs.append(img)

        # create the output directory if not present already
        if not os.path.exists(f"{dir}/predicted"):
            os.makedirs(f"{dir}/predicted")

        # predict the images
        for img in predict_imgs:
            predicted = np.expand_dims(img, 0)  # add the batch dimension to the image
            predicted = self.model(predicted, training=False)  # predict the image
            predicted = np.copy(predicted)
            predicted_images.append(predicted)

        # write the images to files
        for i, img in enumerate(predicted_images):
            plt.imsave(f"{dir}/predicted/{i}.png", img[0])
        print(f"Predicted images in the directory '{dir}/predicted'")

    def train(self):

        # if we want to continue training, we load the previously trained model.
        if self.use_best_weights:
            if not os.listdir(self.checkpoint_dir):
                print("No checkpoint found. Please train the model first.")
            else:

                print(f"[{datetime.now().strftime('%H:%M:%S')}] Best weights loaded.")
                self.model.load_weights(os.path.join(os.getcwd(), self.checkpoint_dir))

        self.model.fit(
            self.train_img_generator,
            validation_data=self.test_img_generator,
            epochs=self.epochs,
            callbacks=self.callbacks_builder(),
            use_multiprocessing=True,
            workers=4,
        )


if __name__ == "__main__":

    args = dict(
        epochs=50,
        batch_size=2,
        patience=5,
        data="demo_set",
        model_path="demo_set",
        continue_training=False,
        early_stopping=True,
        checkpoints=False,
        train=False,
        test=False,
        save_after_train=True,
        epoch_visualization=True,
        tensorboard=True,
        wandb=False,
    )
    args = SimpleNamespace(**args)

    model = DeblurModel(args)
    model.build(256).train()
