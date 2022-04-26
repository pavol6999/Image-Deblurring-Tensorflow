import os
from argparse import ArgumentParser, ArgumentTypeError
import sys


ROOT_DIR = os.getcwd()

from trainer import Trainer


def validate_args(parser):
    args = parser.parse_args()
    args.wandb = False
    print(args)
    if (args.train or args.test) and not args.data:
        parser.error("You must specify folder containing train or test data")
    if (args.test or args.visualize) and (
        args.save_after_train
        or args.checkpoints
        or args.early_stopping
        or args.patience != 5
        or args.tensorboard
        or args.checkpoint_dir
    ):
        parser.error("You can't use checkpoint callback in test or visualize mode")
    if (args.test or args.visualize) and (args.batch_size or args.epochs):
        parser.error("You can't use batch or epochs in test or visualize mode")

    if (args.wandb_project or args.wandb_entity) and not args.wandb_api_key:
        parser.error("You must specify wandb api key if you want to use wandb")
    elif args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        args.wandb = True
    return args


def parse_arguments(args):
    def _num_limit(x):
        x = int(x)
        if x < 1:
            raise ArgumentTypeError("Minimum size is 1")
        return x

    def _valid_path(path):
        path = str(path)
        if not os.path.exists(path):
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
    wandb_group.add_argument(
        "--sweep",
        type=str,
        help="Run in sweep mode. Defined by sweep file. Any other training parameters inferred from command line arguments will be ignored",
    )

    core_group = parser.add_argument_group("CORE", "Arguments for running the neural network")

    phase = core_group.add_mutually_exclusive_group(required=False)
    phase.add_argument(
        "--train", action="store_true", default=True, help="Train the neural network"
    )
    phase.add_argument("--test", action="store_true", help="Test the neural network")
    phase.add_argument("--visualize", action="store_true", help="Visualize the neural network")

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
        "--continue_training",
        required=False,
        action="store_true",
        help="Continue training from the last checkpoint",
    )

    core_group.add_argument(
        "--model_path",
        required=False,
        type=_valid_path,
        help="Path to the model to be loaded in prediction mode. Defaults to None",
        default=None,
    )
    core_group.add_argument("--epoch_visualization", required=False, action="store_true")
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
    callback_group.add_argument(
        "--save_after_train", required=False, action="store_true", help="Save model after training"
    )

    args_out = validate_args(parser)
    return args_out


if __name__ == "__main__":
    args = parse_arguments(os.sys.argv[1:])

    if args.train:
        trainer = Trainer(args)
        if not args.sweep:
            trainer.train()
        else:
            trainer.train_sweep()
