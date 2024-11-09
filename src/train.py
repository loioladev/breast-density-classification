import logging
import os

import torch
import yaml
from torchvision.transforms import v2

from src.datasets.dataloader import ImageDataset, get_dataframe, get_dataloader
from src.utils.config import set_device, set_seed, ConfigManager
from src.utils.logging import CSVLogger
from src.utils.plotting import visualize_dataloader
from src.models import ModelFactory

logger = logging.getLogger()


def main(args: dict) -> None:
    """
    Training function for the script

    :param args: Dictionary containing the parameters for the training
    """
    # -- set device
    device = set_device()
    logger.info(f"Device set to {device}")

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- SEED
    seed = args["meta"]["seed"]
    set_seed(seed)
    logger.info(f"Seed set to {seed} for reproducible results")

    # -- FOLDERS
    # TODO: create folder with the experiment name
    training_folder = args["meta"]["training_folder"]
    os.makedirs(training_folder, exist_ok=True)
    folders = os.listdir(training_folder)
    log_folder = os.path.join(training_folder, "training_" + str(len(folders) + 1))
    os.makedirs(log_folder, exist_ok=True)

    # -- MODEL
    task_type = args["model"]["task_type"]
    model_name = args["model"]["name"]
    model_size = args["model"]["size"]
    pretrained = args["model"]["pretrained"]

    # -- TRAINING
    epochs = args["training"]["epochs"]
    lr = args["training"]["lr"]
    loss_type = args["training"]["loss"]
    schuduler_type = args["training"]["scheduler"]
    optimizer_type = args["training"]["optimizer"]
    # TODO: early stopping

    # -- AUGMENTATION
    height = args["augmentation"]["height"]
    width = args["augmentation"]["width"]

    # TODO: add new transformations, and split in train and test
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((height, width)),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    # -- DATA
    datasets_path = args["data"]["datasets_path"]
    datasets = args["data"]["datasets"]
    sampler = args["data"]["sampler"]
    batch_size = args["data"]["batch_size"]
    workers = args["data"]["workers"]

    train_df, test_df = get_dataframe(datasets, datasets_path, seed)
    logger.info("Datasets loaded")

    train_class = ImageDataset(train_df, transform=transforms)
    dataloader = get_dataloader(train_class, batch_size, sampler, workers=workers)
    logger.info("Dataloader loaded")

    dataloader_visual = os.path.join(log_folder, "dataloader.png")
    visualize_dataloader(
        dataloader, {0: "A", 1: "B", 2: "C", 3: "D"}, dataloader_visual
    )
    logger.info(f"Dataloader visualization saved in {dataloader_visual}")
    # ----------------------------------------------------------------------- #

    # -- save parameters
    dump = os.path.join(log_folder, "params.yaml")
    with open(dump, "w") as f:
        yaml.dump(args, f)
        logger.info(f"Training parameters stored in {dump}")

    # -- make csv logger
    csv_logger = CSVLogger(
        os.path.join(log_folder, "metrics.csv"),
        ("%d", "epoch"),
        ("%.5f", "loss"),
        ("%d", "time (min)"),
    )

    # -- load model
    factory = ModelFactory()
    model = factory.get_model(model_name, task_type, pretrained, model_size=model_size)
    logger.info(f"Model {model_name}{model_size} loaded")
    # TODO: load model from checkpoint

    # -- load optimizer
    optimizer_config = args["optimizer"].get(optimizer_type, 'adamw')
    optimizer_config['weight_decay'] = args["optimizer"]["weight_decay"]
    optimizer = ConfigManager.get_optimizer(model, optimizer_type, optimizer_config)
    logger.info(f"Optimizer {optimizer_type} loaded")

    # -- load scheduler
    scheduler = None
    if schuduler_type:
        scheduler_config = args["scheduler"].get(schuduler_type, {})
        del scheduler_config["metric"]
        scheduler = ConfigManager.get_scheduler(optimizer, schuduler_type, scheduler_config)
        logger.info(f"Scheduler {schuduler_type} loaded")

    # -- load loss
    loss_config = args["loss"].get(loss_type, {})
    loss_weights = train_class.weights()
    loss = ConfigManager.get_loss(loss_type, loss_weights, loss_config)
    logger.info(f"Loss {loss_type} loaded")

    # -- training loop
