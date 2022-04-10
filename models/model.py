from types import SimpleNamespace
import tensorflow as tf
import sys

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

sys.path.append("./")
from blocks import ConvBlock, ConvBlockTranspose
from data.DataGenerator import DataGenerator


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

        self.use_best_weights = args.use_best_weights or False

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
                monitor="val_accuracy",
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

        def chck3():
            pass

        early_stopping_callback = early_stopping()
        chckpoint = model_checkpoint("./checkpoints/")
        return [chckpoint, early_stopping_callback]

    def build(self, patch_size, num_layers=3):
        print("Building model...")

        input = tf.keras.layers.Input((patch_size, patch_size, 3))
        # downsample
        layer1, out_concat = ConvBlock(input, "relu", 64, block_name="Conv_Level_1", padding="same")
        layer2, out_concat2 = ConvBlock(
            layer1, "relu", 128, block_name="Conv_Level_2", padding="same"
        )
        layer3, out_concat3 = ConvBlock(
            layer2, "relu", 256, block_name="Conv_Level_3", padding="same"
        )

        # hrdlo

        # Block 5
        conv_throttle = tf.keras.layers.Conv2D(512, (3, 3), padding="same", name="block5_conv1")(
            layer3
        )
        drop_throttle = tf.keras.layers.Dropout(0.3)(conv_throttle)
        conv_throttle2 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", name="block5_conv2")(
            drop_throttle
        )

        # upsample
        layer_up = ConvBlockTranspose(
            conv_throttle2, out_concat3, "relu", 256, padding="same", block_name="Conv_Up_Level_3"
        )
        layer_up2 = ConvBlockTranspose(
            layer_up, out_concat2, "relu", 128, padding="same", block_name="Conv_Up_Level_2"
        )
        layer_up3 = ConvBlockTranspose(
            layer_up2, out_concat, "relu", 64, padding="same", block_name="Conv_Up_Level_1"
        )

        # output
        output = tf.keras.layers.Conv2D(3, (1, 1), activation="sigmoid")(layer_up3)

        self.model = tf.keras.Model(input, output, name="Deblur")
        return self

    def visualize(self, out_path):
        if not (out_path.endswith(".png") or out_path.endswith(".jpg")):
            out_path += ".png"
        tf.keras.utils.plot_model(self.model, to_file=out_path, show_shapes=True)
        print("Model visualization complete. Output path: " + out_path)

    def train(self):
        train_args = SimpleNamespace(
            shuffle=True,
            seed=1,
            data_path="training_set/GOPR0372_07_00",
            channels=3,  # rgb
            batch_size=2,
            mode="train",
            repeat=1,
        )
        test_args = SimpleNamespace(
            shuffle=True,
            seed=1,
            data_path="training_set/GOPR0384_11_00",
            channels=3,  # rgb
            batch_size=2,
            mode="test",
            repeat=1,
        )

        train_img_generator = DataGenerator(train_args)
        test_img_generator = DataGenerator(test_args)

        model = self.model
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        if self.use_best_weights:
            if not os.listdir(self.checkpoint_dir):
                print("No checkpoint found. Please train the model first.")
            else:

                print("Best weights loaded.")
                model.load_weights(os.path.join(os.getcwd(), self.checkpoint_dir))

        model.fit(
            train_img_generator(),
            validation_data=test_img_generator(),
            epochs=3,
            callbacks=self.callbacks_builder(),
        )


if __name__ == "__main__":

    args = dict(
        epochs=10,
        batch_size=5,
        model_path="model_path",
        checkpoint_dir="./checkpoints/",
        use_best_weights=True,
    )
    args = SimpleNamespace(**args)

    model = DeblurModel(args)
    model.build(256).train()
