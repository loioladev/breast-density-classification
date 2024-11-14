import logging
import os
import time

import yaml

from src.datasets.dataloader import (
    ImageDataset,
    cross_validation,
    get_dataframe,
    get_dataloader,
)
from src.helper import get_transformations, Training
from src.models import ModelFactory
from src.utils.config import ConfigManager, set_device, set_seed
from src.utils.logging import CSVLogger, create_folder
from src.utils.plotting import visualize_dataloader

logger = logging.getLogger()


def main(args: dict) -> None:
    """
    Training function for the script

    :param args: Dictionary containing the parameters for the training
    """
    # -- set device
    device = set_device()
    logger.info(f"Device set to {device}")
    id_to_label = {0: "A", 1: "B", 2: "C", 3: "D"}

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    seed = args["meta"]["seed"]
    task_type = args["meta"]["task_type"]
    training_folder = args["meta"]["training_folder"]
    experiment_name = args["meta"]["experiment_name"]
    checkpoint_dir = args["meta"]["checkpoint_dir"]
    kfolds = args["meta"]["kfolds"]
    metric_types = args["meta"]["metrics"]["types"]
    metric_reduction = args["meta"]["metrics"]["reduction"]

    # -- MODEL
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

    # -- DATA
    datasets_path = args["data"]["datasets_path"]
    datasets = args["data"]["datasets"]
    sampler = args["data"]["sampler"]
    batch_size = args["data"]["batch_size"]
    workers = args["data"]["workers"]
    # ----------------------------------------------------------------------- #

    # -- configure seed
    set_seed(seed)
    logger.info(f"Seed set to {seed} for reproducible results")

    # -- create training folder
    log_folder = create_folder(training_folder, experiment_name)

    # -- save parameters
    dump = os.path.join(log_folder, "params.yaml")
    with open(dump, "w") as f:
        yaml.dump(args, f)
        logger.info(f"Training parameters stored in {dump}")

    # -- transformations
    transformations, target_transformations = get_transformations((height, width))

    # -- load datasets
    train_df, test_df = get_dataframe(datasets, datasets_path, seed)
    train_df = cross_validation(train_df, seed, kfolds, id_to_label)
    logger.info("Datasets loaded")

    # -- check dataloader
    train_class = ImageDataset(train_df, transform=transformations["train"])
    dataloader = get_dataloader(train_class, batch_size, sampler, workers=workers)
    visualize_dataloader(dataloader, id_to_label, log_folder)

    # -- load model
    factory = ModelFactory()
    model = factory.get_model(model_name, task_type, pretrained, model_size=model_size)
    model = model.to(device)
    logger.info(f"Model {model_name}{model_size} loaded")
    # TODO: load model from checkpoint

    # -- load optimizer
    optimizer_config = args["optimizer"].get(optimizer_type, "adamw")
    optimizer_config["weight_decay"] = args["optimizer"]["weight_decay"]
    optimizer = ConfigManager.get_optimizer(model, optimizer_type, optimizer_config)
    logger.info(f"Optimizer {optimizer_type} loaded")

    # -- load scheduler
    scheduler = None
    if schuduler_type:
        scheduler_config = args["scheduler"].get(schuduler_type, {})
        del scheduler_config["metric"]
        scheduler = ConfigManager.get_scheduler(
            optimizer, schuduler_type, scheduler_config
        )
        logger.info(f"Scheduler {schuduler_type} loaded")

    # -- load loss
    loss_config = args["loss"].get(loss_type, {})
    loss_weights = train_class.weights()
    loss = ConfigManager.get_loss(loss_type, loss_weights, loss_config)
    logger.info(f"Loss {loss_type} loaded")

    # -- load metrics
    metric_args = {"task": task_type}
    if task_type == "multiclass":
        metric_args["num_classes"] = len(id_to_label)
        metric_args["average"] = metric_reduction
    metrics = ConfigManager.get_metrics(metric_types, metric_args)

    # ----------------------------------------------------------------------- #
    #  BINARY MODEL TRAINING
    # ----------------------------------------------------------------------- #
    since_binary = time.time()
    targets = train_df["target"].unique().tolist()
    targets = sorted(targets)
    for target in targets:
        logger.info(f"Starting training for target {target}")

        # -- create a binary dataframe for the target
        binary_df = train_df.copy()
        binary_df["target"] = binary_df["target"].apply(lambda x: 1 if x == target else 0)

        # -- initialize folder
        binary_folder = os.path.join(log_folder, f"{target}")
        os.makedirs(binary_folder, exist_ok=True)
        logger.debug(f"Binary folder created at {binary_folder}")

        # -- initialize variables
        since = time.time()
        model_start = model.state_dict()
        optimizer_start = optimizer.state_dict()
        scheduler_start = scheduler.state_dict() if scheduler else None

        for fold in range(kfolds):
            logger.info(f"Training of fold {fold+1}/{kfolds}")

            # -- reset objects
            model.load_state_dict(model_start)
            optimizer.load_state_dict(optimizer_start)
            if scheduler:
                scheduler.load_state_dict(scheduler_start)

            # -- create dataloaders
            fold_train_df = binary_df[binary_df["fold"] != fold]
            fold_val_df = binary_df[binary_df["fold"] == fold]
            train_class = ImageDataset(fold_train_df, transformations["train"], target_transformations["train"])
            val_class = ImageDataset(fold_val_df, transformations["val"], target_transformations["val"])
            train_loader = get_dataloader(train_class, batch_size, sampler, workers=workers)
            val_loader = get_dataloader(val_class, batch_size, workers=workers)
            dataloaders = {"train": train_loader, "val": val_loader}
            logger.debug("Dataloaders created with sizes")
            for key, value in dataloaders.items():
                logger.debug(f"{key}: {len(value.dataset)}")

            folder_path = os.path.join(binary_folder, f"fold_{fold}")
            os.makedirs(folder_path, exist_ok=True)

            # -- make csv logger
            csv_logger = CSVLogger(
                folder_path,
                ("%d", "epoch"),
                ("%.5f", "loss"),
                ("%d", "time"),
                *[("%.2f", m) for m in metric_types if m not in ["confusion"]],
            )

            # -- run training
            training = Training(
                model,
                csv_logger,
                loss,
                optimizer,
                scheduler,
                folder_path,
            )
            training.train(epochs, metrics, dataloaders)
        logger.info(f"Training of {target} model completed in {time.time() - since:.2f}s")
    logger.info(f"Binary training completed in {time.time() - since_binary:.2f}s")