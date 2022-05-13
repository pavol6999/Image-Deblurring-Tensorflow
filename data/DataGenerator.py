import os
import random
from matplotlib import docstring, pyplot as plt
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from types import SimpleNamespace

from tensorflow.keras.layers import GaussianNoise

# training_set/GOPR0372_07_00


class DataGenerator:

    """DataGenerator class provides data generation for model training and validation,
    returning a tensorflow dataset on `__call__`method. Dataset consists of infinite patches from GoPro
    Dataset of size `patch_size`.
    """

    def __init__(self, args):

        self.args = args or None
        self.channels = 3
        self.patch_size = 256
        self.width = 1280
        self.height = 720
        self.shuffle = args.shuffle  # or False
        self.noise = args.noise
        self.flip = args.flip
        self.seed = args.seed  # or 1
        self.data_path = args.data_path
        self.channels = args.channels  # or 3
        self.batch_size = args.batch_size  # or 1
        self.mode = args.mode  # or "train"
        self.repeat = args.repeat or 1
        # defines the number of times the model will see the total number of images

        self.data_path = args.data_path
        assert os.path.isdir(self.data_path), "The data path is not a valid directory"

        # this is the main attribute of the DataGenerator class, it holds the load images
        self.dataset = self.load_dataset(
            self.data_path,
            self.batch_size,
            self.shuffle,
            self.seed,
            self.height,
            self.width,
            self.mode,
            self.channels,
        )

        # as this dataset is to this point infinite, we need to set a cardinality to this dataset, so it does not loop forever.
        # another way to prevent this infinite looping is by setting training_steps in the model.fit() method.
        self.dataset = self.dataset.apply(
            tf.data.experimental.assert_cardinality(self.dataset_size * self.repeat)
        )

    # on calling DataGenerator() object, return our dataset
    def __call__(self, *args, **kwds):
        """

        Returns:
            dataset: tensorflow dataset of two image pair patches
        """
        return self.dataset

    def __len__(self):

        return self.dataset_size * self.repeat

    @staticmethod
    def dimension_modifier(*imgs):
        """Modifier method, that modifies the dimension of the input images to match the model input shape.

        Returns:
            `tuple`: image pair of blur and sharp image
        """
        img1, img2 = imgs
        return tf.squeeze(img1, axis=0), tf.squeeze(img2, axis=0)

    # https://stackoverflow.com/questions/43382045/keras-realtime-augmentation-adding-noise-and-contrast
    @staticmethod
    def add_noise(*imgs):
        """Add random gaussian noise to an image

        Returns:
            `tuple`: image pair of blur and sharp image

        """

        blur, sharp = imgs
        sample = GaussianNoise(0.05)
        noisey = sample(blur, training=True)
        noisey = tf.clip_by_value(noisey, 0, 1, name="clip_noise")
        return noisey, sharp

    @staticmethod
    def create_patches(blur_image, sharp_img, patch_size=256):
        """_summary_
        Args:
            sharp_img (Image `Tensor`): _description_
            blur_image (Image `Tensor`): _description_
            patch_size (`int`): _description_
        Returns:
            Image pair of blur and sharp image patches
        """
        stack = tf.stack([blur_image, sharp_img], axis=0)

        patches = tf.image.random_crop(stack, size=[2, patch_size, patch_size, 3])
        return patches[0], patches[1]

    def load_dataset(
        self,
        path: str,
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int = 1,
        height: int = 720,
        width: int = 1280,
        mode: str = "train",
        channels: int = 3,
    ):
        """
        From the data path, load the dataset using 2 ImageDataGenerators (one for blur and one for sharp `ground truth` image)

        """

        gen1, gen2 = DataGenerator.create_generators(
            path,
            shuffle,
            seed,
            height,
            width,
            mode,
            channels,
            self.flip,
        )
        dataset = self.combine_generators_into_dataset(gen1, gen2).batch(batch_size)

        self.dataset_size = len(gen1.filenames) // batch_size

        return dataset

    @staticmethod
    def create_generators(
        path: str,
        shuffle: bool = True,
        seed: int = 1,
        height: int = 720,
        width: int = 1280,
        mode: str = "train",
        channels: int = 3,
        flip=False,
    ):
        """_summary_

        Args:
            path (str): path to the dataset
            shuffle (bool, optional): Shuffles the dataset. Defaults to False.
            seed (int, optional): Defaults to 1.
            height (int, optional): Height of images from the dataset. Defaults to 720.
            width (int, optional): Width of images from the dataset. Defaults to 1280.
            mode (str, optional): `train` or `test`. Defaults to "train".
            channels (int, optional): _description_. Defaults to 3.

        Returns:
            _type_: _description_
        """
        if mode == "train":

            # initialize our image generators for blur and sharp images. This is the train
            BlurredDataGenerator = ImageDataGenerator(
                rescale=1.0 / 255.0, horizontal_flip=flip, vertical_flip=flip
            )
            SharpDataGenerator = ImageDataGenerator(
                rescale=1.0 / 255.0, horizontal_flip=flip, vertical_flip=flip
            )
        else:

            BlurredDataGenerator = ImageDataGenerator(
                rescale=1.0 / 255.0,
            )
            SharpDataGenerator = ImageDataGenerator(
                rescale=1.0 / 255.0,
            )
        BlurImgGenerator = BlurredDataGenerator.flow_from_directory(
            f"{path}/blur",
            seed=seed,
            class_mode=None,
            color_mode="rgb" if channels == 3 else "grayscale",
            batch_size=1,
            shuffle=shuffle,
            target_size=(height, width),
        )

        SharpImgGenerator = SharpDataGenerator.flow_from_directory(
            f"{path}/sharp",
            seed=seed,
            class_mode=None,
            color_mode="rgb" if channels == 3 else "grayscale",
            batch_size=1,
            shuffle=shuffle,
            target_size=(height, width),
        )

        assert len(BlurImgGenerator.filenames) == len(
            SharpImgGenerator.filenames
        ), "Mismatching number of image pairs"

        return BlurImgGenerator, SharpImgGenerator

    @staticmethod
    def preprocess_dataset(data, noise):
        """Preprocessing function, that first pops out the batch dimension,
        then creates uniform patches at the same location of the image pairs
        and if `noise` is True, then adds a Gaussian Noise to the blurred crop of the image pair

        Args:
            data (`FlatMapDataset | DatasetV1Adapter`): Tensorflow dataset of image pairs

        Returns:
            FlatMapDataset | DatasetV1Adapter: Tensorflow dataset of image pair patches
        """
        dataset = data.map(DataGenerator.dimension_modifier, num_parallel_calls=tf.data.AUTOTUNE)

        # dataset = dataset.map(get_patches)
        dataset = dataset.map(
            DataGenerator.create_patches,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        # dataset = dataset.unbatch()
        if noise:
            dataset = dataset.map(
                DataGenerator.add_noise,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        return dataset

    def combine_generators_into_dataset(self, gen1, gen2):
        """Combines two generators into one preprocessed dataset.

        Args:
            gen1 (ImageDataGenerator): ImageDataGenerator for blur images
            gen2 (ImageDataGenerator): ImageDataGenerator for sharp images

        Returns:
            `Tensorflow Dataset`
        """

        combined_generator = zip(gen1, gen2)
        dataset = tf.data.Dataset.from_generator(
            lambda: combined_generator,
            output_signature=(
                tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.float32),
            ),
        )
        dataset = DataGenerator.preprocess_dataset(dataset, self.noise)
        return dataset


if __name__ == "__main__":
    args = dict(
        shuffle=False,
        seed=1,
        data_path="training_set/train",
        batch_size=16,
        mode="train",
        noise=True,
        channels=3,
        flip=True,
        repeat=1,
    )
    args = SimpleNamespace(**args)
    data_generator = DataGenerator(args)

    print(
        f"There are a total of {len(data_generator)} batches of {data_generator.batch_size} image pairs"
    )
    print(
        f"Total of {len(data_generator)*data_generator.batch_size} image pairs will be fed to the model"
    )
    for blur, sharp in data_generator().take(20):
        f, axarr = plt.subplots(2)
        axarr[0].imshow(blur[0].numpy())
        axarr[1].imshow(sharp[0].numpy())

        plt.show()

        input()
