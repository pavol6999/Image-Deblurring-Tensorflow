import click
import wandb
import yaml
from model.model import DeblurModel
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Dataset
from azureml.core import ScriptRunConfig
from azureml.core import Experiment
from azureml.core.environment import Environment


class Trainer:
    def __init__(self, args):
        self.args = args

    # function that will call the model.predict()
    def predict(self):
        model = DeblurModel(self.args)
        model.predict()

    # This method handles the train mode of our model. It asks us if we want to use the azure machine learning studio or if we want to train on the local machine
    def train(self):

        if click.confirm("Do you wish to train on Azure ML?", default=False):
            print("Parameters loaded from command line will be ignored. Except wandb api key")
            ws = Workspace.from_config(
                path="azure_config.json",  # This is the path to your azure config file
                auth=InteractiveLoginAuthentication(),
            )
            print(ws.name, ws.location, ws.resource_group, sep="\t")
            dataset = Dataset.get_by_name(workspace=ws, name="GoPro_v2")

            config = ScriptRunConfig(
                source_directory="./",
                script="train_azure.py",
                arguments=[
                    "--wandb_api_key",
                    self.args.wandb_api_key,  # Required for wandb. this is the api key. if you want to use a different api key, this must be changed.
                    "--data",
                    dataset.as_named_input("GoPro_v2").as_mount(
                        "/tmp/GoPro_v2"
                    ),  # This is important how to mount dataset (GoPro_v2) from DataStore
                ],  # This is important how to mount dataset from DataStore
                compute_target="K80x4-Krajkovic",  # compute target
            )  # Compute target is your created compute cluster

            experiment = Experiment(workspace=ws, name="Deblur_v2")
            env = Environment.get(workspace=ws, name="krajkovic-env")
            config.run_config.environment = env
            run = experiment.submit(config)

            aml_url = run.get_portal_url()
            print("Submitted to compute cluster. Click link below")
            print("")
            print(aml_url)

        else:
            model = DeblurModel(self.args)
            model.build(256).train()

    def train_sweep(self):
        if click.confirm("Do you wish to train on Azure ML?", default=False):
            print("Parameters loaded from command line will be ignored. Except wandb api key")
            ws = Workspace.from_config(
                path="azure_config.json",
                auth=InteractiveLoginAuthentication(
                    #  tenant_id="5dbf1add-202a-4b8d-815b-bf0fb024e033" # This is the tenant id of your azure workspace, if you have one azure workspace account, this mustnt be specified
                ),
            )
            print(ws.name, ws.location, ws.resource_group, sep="\t")
            dataset = Dataset.get_by_name(workspace=ws, name="GoPro_v2")

            config = ScriptRunConfig(
                source_directory="./",
                script="train_sweep.py",
                arguments=[
                    "--wandb_api_key",
                    self.args.wandb_api_key,  # Required for wandb. this is the api key. if you want to use a different api key, this must be changed.
                    "--data",
                    dataset.as_named_input("GoPro_v2").as_mount("/tmp/GoPro_v2"),
                ],  # This is important how to mount dataset from DataStore
                compute_target="K80x4-Krajkovic",
            )  # Compute target is your created compute cluster
            experiment = Experiment(workspace=ws, name="Deblur_v2")
            env = Environment.get(workspace=ws, name="krajkovic-env")
            config.run_config.environment = env
            run = experiment.submit(config)

            aml_url = run.get_portal_url()
            print("Submitted to compute cluster. Click link below")
            print("")
            print(aml_url)

        else:
            raise Exception(
                "Due to the lack of computing power. You can train sweeps only on Azure"
            )
