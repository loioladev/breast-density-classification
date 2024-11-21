import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from src.datasets.dataloader import (
    ImageDataset,
    cross_validation,
    get_dataframe,
    get_dataloader,
)
from src.helper import BinaryModelTester, BinaryModelTrainer
from src.models import ModelFactory
from src.transforms import get_transformations
from src.utils.config import ConfigManager, set_device, set_seed
from src.utils.logging import CSVLogger, convert_time, create_folder
from src.utils.plotting import plot_confusion_matrix, visualize_dataloader

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

    # -- initialize binary classification
    binary_classification = BinaryClassification(
        log_folder,
        model,
        loss,
        optimizer,
        scheduler,
        transformations,
        target_transformations,
        batch_size,
        workers,
        sampler,
    )

    # -- train models
    binary_classification.train_models(train_df, kfolds, epochs, metrics)

    # -- test models
    binary_classification.test_models(test_df)


class BinaryClassification:
    def __init__(
        self,
        folder: str,
        model,
        criterion,
        optimizer,
        scheduler,
        transforms,
        target_transforms,
        batch_size,
        workers,
        sampler,
    ) -> None:  # TODO: ABC
        self.run_folder = Path(folder)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.batch_size = batch_size
        self.workers = workers
        self.sampler = sampler
        self.device = set_device()

    def load_data(self, dataframe: pd.DataFrame, fold: int) -> dict:  # TODO: ABC
        """
        Load dataloaders for training and validation, according to folds for
        cross validation.

        :param dataframe: The dataframe to split
        :param fold: The fold to use for validation
        :return dataloaders: The dataloaders for training and validation
        """
        fold_train_df = dataframe[dataframe["fold"] != fold]
        fold_val_df = dataframe[dataframe["fold"] == fold]

        # TODO: log fold distribution

        train_class = ImageDataset(
            fold_train_df, self.transforms["train"], self.target_transforms["train"]
        )
        val_class = ImageDataset(
            fold_val_df, self.transforms["val"], self.target_transforms["val"]
        )

        train_loader = get_dataloader(
            train_class, self.batch_size, self.sampler, workers=self.workers
        )
        val_loader = get_dataloader(val_class, self.batch_size, workers=self.workers)

        dataloaders = {"train": train_loader, "val": val_loader}
        return dataloaders

    def create_logger(self, folder_path: str, metrics: dict) -> CSVLogger:  # TODO: abc
        """
        Create an instance of the CSV logger to store the metrics

        :param folder_path: The path to the folder to store the metrics
        :param metrics: The metrics to store
        :return: The instance of the CSV logger
        """
        logger_metrics = []
        for metric_type in metrics:
            if metric_type in ["confusion"]:
                logger_metrics.append(("%s", metric_type))
                continue
            logger_metrics.append(("%.5f", metric_type))

        csv_logger = CSVLogger(
            folder_path,
            ("%d", "epoch"),
            ("%.5f", "loss"),
            ("%d", "time"),
            *logger_metrics,
        )
        return csv_logger

    def train_models(
        self, dataframe: pd.DataFrame, kfolds: int, epochs: int, metrics: list
    ) -> None:
        """
        Initialize training of the binary classification models

        :param dataframe: The dataframe to use for training
        :param kfolds: The number of folds to use for cross validation
        :param epochs: The number of epochs to train the models
        :param metrics: The MetricCollection to use for evaluation
        """
        since_binary = time.time()

        # -- obtain targets in **ascending order**
        targets = dataframe["target"].unique().tolist()
        targets = sorted(targets)
        for target in targets:
            since_target = time.time()
            logger.info(f"Starting training for target {target}")

            # -- initialize target folder
            binary_folder = Path(self.run_folder) / str(target)
            os.makedirs(binary_folder, exist_ok=True)

            # -- create a binary dataframe for the target
            binary_df = dataframe.copy()
            binary_df["target"] = binary_df["target"].apply(
                lambda x: 1 if x == target else 0
            )
            # TODO: log distribution of target

            # -- store pretrained states
            model_start = self.model.state_dict()
            optimizer_start = self.optimizer.state_dict()
            scheduler_start = self.scheduler.state_dict() if self.scheduler else None

            for fold in range(kfolds):
                logger.info(f"Starting training on fold {fold+1}/{kfolds}")

                # -- create storage of fold
                folder_path = os.path.join(binary_folder, f"fold_{fold}")
                os.makedirs(folder_path, exist_ok=True)

                # -- create dataloaders and logger
                dataloaders = self.load_data(binary_df, fold)
                csv_logger = self.create_logger(folder_path, metrics)

                # -- start fold training
                trainer = BinaryModelTrainer(
                    self.model,
                    self.criterion,
                    self.optimizer,
                    self.scheduler,
                    csv_logger,
                    folder_path,
                )
                trainer.train(epochs, metrics, dataloaders)

                # -- reset models to pretrained version
                self.model.load_state_dict(model_start)
                self.optimizer.load_state_dict(optimizer_start)
                if self.scheduler:
                    self.scheduler.load_state_dict(scheduler_start)

            logger.info(
                f"Models of target {target} trained in {convert_time(time.time() - since_target)}"
            )
        logger.info(
            f"Binary models classification training completed in {convert_time(time.time() - since_binary)}"
        )

    def test_models(self, dataframe: pd.DataFrame):
        """"""
        logger.info("Start testing binary models")
        results = []
        for target in dataframe["target"].unique():
            logger.info(f"Starting testing for target {target}")

            # -- create a binary dataframe for the target
            binary_df = dataframe.copy()
            binary_df["target"] = binary_df["target"].apply(
                lambda x: 1 if x == target else 0
            )

            # -- load dataloader
            binary_class = ImageDataset(
                binary_df, self.transforms["val"], self.target_transforms["val"]
            )
            dataloader = get_dataloader(
                binary_class, self.batch_size, workers=self.workers
            )

            # -- initialize tester instance
            target_folder = self.run_folder / str(target)
            binary_tester = BinaryModelTester(
                self.model, target_folder, dataloader, self.device
            )

            # -- obtain predictions for the target
            target_predictions = binary_tester.test()
            threshold = binary_tester.find_optimal_threshold(target_predictions)
            results.append((target_predictions, threshold))
            logger.info(
                f"Test in target {target} done. Threshold {threshold:.2f} obtained"
            )
        logger.info("Binary models testing completed")

        # -- calculate the metrics
        best_predictions = self.select_best_predictions(dataframe, results)
        ground_truth = dataframe["target"].tolist()
        report = classification_report(
            ground_truth,
            best_predictions,
            target_names=["A", "B", "C", "D"],
            digits=3,
            zero_division=0,
        )
        logger.info(report)

        # -- calculate confusion matrix
        cm = confusion_matrix(ground_truth, best_predictions)
        plot_confusion_matrix(cm, self.run_folder)

    def select_best_predictions(
        self, dataframe: pd.DataFrame, values: list[float]
    ) -> list[int]:
        """
        Select best predictions based on the threshold values

        :param values: The list of predictions and thresholds
        """
        # -- initialize variable
        n_samples = len(dataframe)
        n_classes = len(values)
        best_predictions = np.full(n_samples, -1, dtype=int)

        for i in range(n_samples):
            # -- extract predictions and thresholds for this sample
            sample_predictions = np.array([values[j][0][i] for j in range(n_classes)])
            sample_thresholds = np.array([values[j][1] for j in range(n_classes)])

            # -- find indices where prediction meets or exceeds threshold
            valid_indices = np.where(sample_predictions >= sample_thresholds)[0]

            if len(valid_indices) > 0:
                # -- if any predictions meet threshold, take the last (highest index)
                best_predictions[i] = valid_indices[-1]
            else:
                # -- if no predictions meet threshold, find the closest to threshold
                threshold_distances = sample_thresholds - sample_predictions
                best_predictions[i] = np.argmin(threshold_distances)

        return best_predictions
