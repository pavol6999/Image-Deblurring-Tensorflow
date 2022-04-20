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
from blocks import ConvBlock, ConvBlockTranspose
from data.DataGenerator import DataGenerator
from utils.callbacks import PredictImageAfterEpoch, SaveTrainedModel
from utils.metrics import PSNR, SSIM
from utils.losses import MeanGradientError, SSIMLoss, SobelLoss


class DeblurModel:
    """
    Base class for deblur model architecture based on the U-Net architecture.
    """

    def __init__(self, args):
        self.args = args or None
        self.epochs = args.epochs  # or 10
        self.batch_size = args.batch_size  # or 1
        self.model = (
            args.model_path
        )  # TODO make model loading process `load_model(model_path) or build_model()`
        self.checkpoint_dir = args.checkpoint_dir or None

        self.use_best_weights = args.continue_training or False
        self.early_stopping = args.early_stopping or False
        self.save_trained_model = args.save_trained_model or False
        self.checkpoint = args.checkpoint or False
        self.epoch_visualization = args.epoch_visualization or True
        self.tensorboard = args.tensorboard or False
        self.wandb = args.wandb or True
        train_args = SimpleNamespace(
            shuffle=True,
            seed=1,
            data_path="training_set/train",
            channels=3,  # rgb
            batch_size=4,
            mode="train",
            repeat=1,
        )
        test_args = SimpleNamespace(
            shuffle=True,
            seed=1,
            data_path="training_set/test",
            channels=3,  # rgb
            batch_size=4,
            mode="test",
            repeat=1,
        )

        self.train_img_generator = DataGenerator(train_args)
        self.test_img_generator = DataGenerator(test_args)

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
                monitor="val_psnr",
                mode="max",
                verbose=20,
                save_best_only=True,
            )
            return model_checkpoint_callback

        def early_stopping():
            return tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=2,
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
                generator=self.train_img_generator(),
                validation_steps=10,
                predictions=10,
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
            callbacks.append(model_checkpoint("./checkpoints/"))

        if self.save_trained_model:
            callbacks.append(SaveTrainedModel("deblurmodel", 1))

        if self.epoch_visualization:
            image_pair = self.train_img_generator().take(1)
            callbacks.append(PredictImageAfterEpoch(image_pair))

        if self.wandb:
            callbacks.append(wandb_callback())
        return callbacks

    def build(self, patch_size, num_layers=3):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Building model...")

        input = tf.keras.layers.Input((patch_size, patch_size, 3))
        # downsample
        layer1 = ConvBlock(input, "relu", 64, block_name="Conv_Level_1", padding="same")
        maxpooling_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(
            layer1
        )

        layer2 = ConvBlock(maxpooling_1, "relu", 128, block_name="Conv_Level_2", padding="same")
        maxpooling_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(
            layer2
        )

        layer3 = ConvBlock(maxpooling_2, "relu", 256, block_name="Conv_Level_3", padding="same")
        maxpooling_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(
            layer3
        )

        layer4 = ConvBlock(maxpooling_3, "relu", 512, block_name="Conv_Level_4", padding="same")
        maxpooling_4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(
            layer4
        )

        # hrdlo

        # Block 5
        conv_throttle = tf.keras.layers.Conv2D(1024, (3, 3), padding="same", name="throttle_conv1")(
            maxpooling_4
        )
        drop_throttle = tf.keras.layers.Dropout(0.3)(conv_throttle)
        conv_throttle2 = tf.keras.layers.Conv2D(
            1024, (3, 3), padding="same", name="throttle_conv2"
        )(drop_throttle)
        conv_throttle3 = tf.keras.layers.Conv2D(
            1024, (3, 3), padding="same", name="throttle_conv3"
        )(conv_throttle2)
        # upsample

        layer_up = ConvBlockTranspose(
            conv_throttle3, layer4, "relu", 512, padding="same", block_name="Conv_Up_Level_4"
        )

        layer_up1 = ConvBlockTranspose(
            layer_up, layer3, "relu", 256, padding="same", block_name="Conv_Up_Level_3"
        )
        layer_up2 = ConvBlockTranspose(
            layer_up1, layer2, "relu", 128, padding="same", block_name="Conv_Up_Level_2"
        )
        layer_up3 = ConvBlockTranspose(
            layer_up2, layer1, "relu", 64, padding="same", block_name="Conv_Up_Level_1"
        )

        # output
        output = tf.keras.layers.Conv2D(3, (1, 1), activation="sigmoid")(layer_up3)

        self.model = tf.keras.Model(input, output, name="Deblur")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Finished Building")

        return self

    def visualize(self, out_path):
        if not (out_path.endswith(".png") or out_path.endswith(".jpg")):
            out_path += ".png"
        tf.keras.utils.plot_model(self.model, to_file=out_path, show_shapes=True)
        print("Model visualization complete. Output path: " + out_path)

    def train(self):

        model = self.model
        loss = {"ssim": SSIMLoss, "sobel": SobelLoss}
        model.compile(
            optimizer="adam",
            loss=SSIMLoss,
            metrics=["accuracy", PSNR, SSIM],
        )

        if self.use_best_weights:
            if not os.listdir(self.checkpoint_dir):
                print("No checkpoint found. Please train the model first.")
            else:

                print(f"[{datetime.now().strftime('%H:%M:%S')}] Best weights loaded.")
                model.load_weights(os.path.join(os.getcwd(), self.checkpoint_dir))

        model.fit(
            self.train_img_generator(),
            validation_data=self.test_img_generator(),
            epochs=50,
            callbacks=self.callbacks_builder(),
            use_multiprocessing=True,
            workers=4,
        )


if __name__ == "__main__":

    wandb.config = {"learning_rate": 0.001, "epochs": 50, "batch_size": 4}

    wandb.init(entity="xkrajkovic", project="bp_deblur", sync_tensorboard=False)

    args = dict(
        epochs=10,
        batch_size=5,
        model_path="model_path",
        checkpoint_dir="./checkpoints/",
        continue_training=False,
        early_stopping=False,
        checkpoint=True,
        save_trained_model=True,
        epoch_visualization=True,
        tensorboard=False,
        wandb=True,
    )
    args = SimpleNamespace(**args)

    model = DeblurModel(args)
    model.build(256).train()
