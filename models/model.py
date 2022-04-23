from datetime import datetime
import math
from types import SimpleNamespace
import tensorflow as tf
import sys
import wandb
from wandb.keras import WandbCallback
import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

sys.path.append("./")
from models.blocks import ConvBlock, ConvBlockTranspose
from data.DataGenerator import DataGenerator
from utils.callbacks import PredictImageAfterEpoch, SaveTrainedModel
from utils.metrics import PSNR, SSIM
from utils.losses import MeanGradientError, SSIM_L1_Loss, SSIMLoss, SobelLoss


class DeblurModel:
    """
    Base class for deblur model architecture based on the U-Net architecture.
    """

    def __init__(self, args, config=None):

        ############## Default config ################
        self.learning_rate = 0.001
        self.optimizer = "adam"
        self.dropout_chance = 0.3
        self.batch_size = 4
        self.kernel_size = 3
        self.loss_function = "SSIMLOSS"
        self.filters = 64
        ##############################################

        self.args = args or None
        self.epochs = args.epochs  # or 10
        self.batch_size = args.batch_size  # or 1
        self.model = (
            args.model_path
        )  # TODO make model loading process `load_model(model_path) or build_model()`
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

        self.train_phase = args.train
        self.test_phase = args.test
        self.visualize_phase = args.visualize
        self.patience = args.patience

        ## Override default config with user config ##
        if config is not None:
            self.learning_rate = config["learning_rate"]  #
            self.optimizer = config["optimizer"]  #
            self.dropout_chance = config["dropout"]  #
            self.batch_size = config["batch_size"]  #
            self.kernel_size = config["kernel_size"]  #
            self.loss_function = config["loss_function"]
            self.filters = config["filters"]  #

        ##############################################
        if self.wandb and config is None:
            wandb.init(entity="xkrajkovic", project="bp_deblur", sync_tensorboard=False)

        if self.train_phase:

            self.train_args = SimpleNamespace(
                shuffle=True,
                seed=1,
                data_path=f"{self.data}/train",
                channels=3,  # rgb
                noise=False,
                flip=True,
                batch_size=self.batch_size,
                mode="train",
                repeat=2,
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

    def load_model(self, model_path):
        """
        Loads the model from the specified path.
        """
        self.model = tf.keras.models.load_model(model_path)

    def save_model(self, out_path, model_name):
        """
        Saves the model to the specified path.
        """
        self.model.save(out_path + model_name)

    def callbacks_builder(self):
        def model_checkpoint(path):

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
            return tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=self.patience,
                verbose=1,
                mode="auto",
                baseline=None,
                restore_best_weights=False,
            )

        def tensorboard_checkpoint_callback():
            log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        def wandb_callback():
            return WandbCallback(
                log_weights=False,
                generator=self.train_img_generator,
                validation_steps=20,
                predictions=20,
                input_type="image",
                output_type="image",
                log_evaluation=True,
                log_evaluation_frequency=1,
            )

        callbacks = []

        if self.early_stopping:
            callbacks.append(early_stopping())
        if self.tensorboard:
            callbacks.append(tensorboard_checkpoint_callback())

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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Building model...")

        mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():

            if self.train_phase:
                options = tf.data.Options()
                options.experimental_distribute.auto_shard_policy = (
                    tf.data.experimental.AutoShardPolicy.OFF
                )

                self.train_img_generator = DataGenerator(self.train_args)().with_options(options)
                self.test_img_generator = DataGenerator(self.test_args)().with_options(options)

            input = tf.keras.layers.Input((None, None, 3))

            # downsample
            layer1 = ConvBlock(
                input,
                "relu",
                self.filters,
                block_name="Conv_Level_1",
                dropout_chance=self.dropout_chance,
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
                dropout_chance=self.dropout_chance,
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
                dropout_chance=self.dropout_chance,
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
                dropout_chance=self.dropout_chance,
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
            drop_throttle = tf.keras.layers.Dropout(self.dropout_chance)(conv_throttle)
            conv_throttle2 = tf.keras.layers.Conv2D(
                self.filters * 16,
                (self.kernel_size, self.kernel_size),
                padding="same",
                name="throttle_conv2",
            )(drop_throttle)
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
                dropout_chance=self.dropout_chance,
                kernel_size=(self.kernel_size, self.kernel_size),
                block_name="Conv_Up_Level_4",
            )

            layer_up1 = ConvBlockTranspose(
                layer_up,
                layer3,
                "relu",
                self.filters * 4,
                padding="same",
                dropout_chance=self.dropout_chance,
                kernel_size=(self.kernel_size, self.kernel_size),
                block_name="Conv_Up_Level_3",
            )
            layer_up2 = ConvBlockTranspose(
                layer_up1,
                layer2,
                "relu",
                self.filters * 2,
                padding="same",
                dropout_chance=self.dropout_chance,
                kernel_size=(self.kernel_size, self.kernel_size),
                block_name="Conv_Up_Level_2",
            )
            layer_up3 = ConvBlockTranspose(
                layer_up2,
                layer1,
                "relu",
                self.filters,
                padding="same",
                dropout_chance=self.dropout_chance,
                kernel_size=(self.kernel_size, self.kernel_size),
                block_name="Conv_Up_Level_1",
            )

            # # output
            output = tf.keras.layers.Conv2D(3, (1, 1), activation="sigmoid")(layer_up3)

            self.model = tf.keras.Model(input, output, name="Deblur")

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

            elif self.loss_function == "SobelLoss":
                self.model.compile(
                    optimizer=optimizer,
                    loss=SobelLoss,
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

    def visualize(self, out_path):
        if not (out_path.endswith(".png") or out_path.endswith(".jpg")):
            out_path += ".png"
        tf.keras.utils.plot_model(self.model, to_file=out_path, show_shapes=True)
        print("Model visualization complete. Output path: " + out_path)

    def train(self):

        if self.use_best_weights:
            if not os.listdir(self.checkpoint_dir):
                print("No checkpoint found. Please train the model first.")
            else:

                print(f"[{datetime.now().strftime('%H:%M:%S')}] Best weights loaded.")
                model.load_weights(os.path.join(os.getcwd(), self.checkpoint_dir))

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
        batch_size=1,
        patience=5,
        data="training_set",
        model_path="model_path",
        continue_training=False,
        early_stopping=True,
        checkpoints=False,
        train=True,
        test=False,
        visualize=False,
        save_after_train=True,
        epoch_visualization=True,
        tensorboard=False,
        wandb=False,
        wandb_api_key="026253717624f7e54ae9c7fdbf1c08b1267a9ec4",
    )
    args = SimpleNamespace(**args)

    model = DeblurModel(args)
    model.build(256).train()
