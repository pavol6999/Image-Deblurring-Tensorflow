from types import SimpleNamespace
import tensorflow as tf
import sys


sys.path.append("./")
from blocks import ConvBlock, ConvBlockTranspose
from DataGenerator import DataGenerator


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

    def chekpoint_builder(self):
        def chck1():
            pass

        def chck2():
            pass

        def chck3():
            pass

        raise NotImplementedError("No checkpoints implemnted yet")

    def build(self, patch_size, num_layers=3):
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

        self.model = tf.keras.Model(input, output)
        return self

    def visualize(self, out_path):
        if not (out_path.endswith(".png") or out_path.endswith(".jpg")):
            out_path += ".png"
        tf.keras.utils.plot_model(self.model, to_file=out_path, show_shapes=True)
        print("Model visualization complete. Output path: " + out_path)

    def train(self):
        train_args = dict(
            shuffle=True,
            seed=1,
            data_path="../training_set/GOPR0372_07_00",
            channels=3,  # rgb
            batch_size=1,
            mode="train",
        )
        train_args = SimpleNamespace(**train_args)
        train_img_generator = DataGenerator(train_args)
        model = self.model
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.summary()
        model.fit(
            train_img_generator(),
            validation_data=train_img_generator(),
            epochs=1,
        )
        print("kkt")


if __name__ == "__main__":

    args = dict(epochs=10, batch_size=5, model_path="model_path")
    args = SimpleNamespace(**args)

    model = DeblurModel(args)
    model.build(256).train()
