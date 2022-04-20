import os
from argparse import ArgumentParser, ArgumentTypeError
from tokenize import Number


ROOT_DIR = os.getcwd()


def parse_arguments(args):
    def _num_limit(x):
        x = int(x)
        if x < 1:
            raise ArgumentTypeError("Minimum size is 1")
        return x

    def _valid_path(path):
        path = str(path)
        if os.path.exists(path):
            raise ArgumentTypeError(f"{path} is not a valid path")
        return path

    parser = ArgumentParser(
        description="U-Net Neural Network for Dynamic Image Deblurring",
        epilog="Pavol Krajkovic, FIIT STU in Bratislava, Image Processing using Deep Learning Methods",
    )
    wandb_group = parser.add_argument_group("WandB", "Arguments for WandB integration")
    wandb_group.add_argument(
        "--wandb_api_key",
        required=False,
        type=str,
        help="Wandb API key to log your experiments",
    )
    wandb_group.add_argument(
        "--wandb_project",
        type=str,
        required=False,
        help="The name of the project where you're sending the new run",
    )
    wandb_group.add_argument(
        "--wandb_entity",
        type=str,
        required=False,
        help="An entity is a username or team name where you're sending runs. This entity must exist before you can send runs there, so make sure to create your account or team in the UI before starting to log runs",
    )

    core_group = parser.add_argument_group("CORE", "Arguments for running the neural network")
    phase = core_group.add_mutually_exclusive_group(required=True)
    phase.add_argument("--train", action="store_true", help="Train the neural network")
    phase.add_argument("--test", action="store_true", help="Test the neural network")

    core_group.add_argument(
        "-e",
        "--epochs",
        type=_num_limit,
        required=False,
        help="Number of epochs to run the training. Defaults to 100",
        default=100,
    )

    core_group.add_argument(
        "--data",
        type=str,
        required=False,
        help="File containing the list of image pairs to be processed. Each line must have a pair of paths leading to a sharp and a blur image. In case of predicting the blur image, the path to the blur image must be specified",
    )

    core_group.add_argument(
        "-b",
        "--batch_size",
        help="Size of the training batch. Defaults to 4",
        type=_num_limit,
        default=4,
        required=False,
    )

    core_group.add_argument(
        "--model_path",
        required=False,
        type=_valid_path,
        help="Path to the model to be loaded in prediction mode. Defaults to None",
        default=None,
    )

    callback_group = parser.add_argument_group("Callbacks")
    callback_group.add_argument(
        "--checkpoint_dir",
        required=False,
        type=_valid_path,
        default=None,
        help="Checkpoint directory path to be used in training mode, where the weights will be stored that can be later used to continue training. Defaults to None",
    )
    callback_group.add_argument(
        "--checkpoints",
        required=False,
        # default=False,
        action="store_true",
        help="Enable checkpoint callback after each epoch. Saves only the best model.",
    )

    callback_group.add_argument(
        "--tensorboard", required=False, action="store_true", help="Enable tensorboard callback"
    )
    callback_group.add_argument(
        "--early_stopping",
        required=False,
        action="store_true",
        help="Enable early stopping callback",
    )
    callback_group.add_argument(
        "--patience", required=False, type=int, default=5, help="Early stopping patience"
    )
    args_out = parser.parse_args()
    return args_out


if __name__ == "__main__":
    args = parse_arguments(os.sys.argv[1:])
    print(args)

    print("kkt")
