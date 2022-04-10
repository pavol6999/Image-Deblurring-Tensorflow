import os
from argparse import ArgumentParser
from data.DataGenerator import DataGenerator

ROOT_DIR = os.getcwd()


def parse_arguments(args):
    parser = ArgumentParser(
        description="U-Net Neural Network for Dynamic Image Deblurring",
        epilog="Pavol Krajkovic, FIIT STU in Bratislava, Image Processing using Deep Learning Methods",
    )

    parser.add_argument(
        "-p",
        "--phase",
        type=str,
        default="train",
        help="Determine the phase of the model to be used (train or test)",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=False,
        help="File containing the list of image pairs to be processed. Each line must have a pair of paths leading to a sharp and a blur image.",
    )
    parser.add_argument(
        "-b", "--batch_size", help="Size of the training batch", type=int, default=16
    )

    args_out = parser.parse_args()
    return args_out


if __name__ == "__main__":
    args = parse_arguments(os.sys.argv[1:])
    print(args)

    generator = DataGenerator(args)
    data = generator.input_generator()

    print("kkt")
