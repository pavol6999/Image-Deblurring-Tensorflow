from argparse import ArgumentParser
import os
from types import SimpleNamespace
import wandb
import yaml

from models.model import DeblurModel


def sweep():
    with wandb.init() as run:
        config = wandb.config
        data_path = os.getenv("DATA_FOLDER")
        idk = SimpleNamespace(
            epochs=20,
            batch_size=16,
            patience=None,
            data=data_path,
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
            wandb_api_key="026253717624f7e54ae9c7fdbf1c08b1267a9ec4",
        )
        model = DeblurModel(idk, config)
        model.build(256).train()


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
    os.environ["DATA_FOLDER"] = args.data
    with open("sweep.yaml", "r") as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            print(parsed_yaml)
            sweep_id = wandb.sweep(parsed_yaml, entity="xkrajkovic", project="bp_deblur")
            count = 10  # number of runs to execute
            wandb.agent(sweep_id, function=sweep, count=count)
        except yaml.YAMLError as exc:
            print(exc)
