from argparse import ArgumentParser
from types import SimpleNamespace
from models.model import DeblurModel

import os

if __name__ == "__main__":

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
    parser.add_argument(
        "--data",
        type=str,
        required=False,
        help="File containing the list of image pairs to be processed. Each line must have a pair of paths leading to a sharp and a blur image. In case of predicting the blur image, the path to the blur image must be specified",
    )
    args = parser.parse_args()

    os.environ["WANDB_API_KEY"] = args.wandb_api_key

    args = SimpleNamespace(
        epochs=100,
        batch_size=4,
        patience=None,
        data=args.data,
        model_path="model_path",
        continue_training=False,
        early_stopping=False,
        checkpoints=False,
        train=True,
        test=False,
        visualize=False,
        save_after_train=True,
        epoch_visualization=True,
        tensorboard=False,
        wandb=True,
        wandb_api_key=args.wandb_api_key,
    )
    model = DeblurModel(args)
    model.build(256).train()
