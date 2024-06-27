"""A collection of experiment tracking utilities."""

import os
import csv
import torch
from contextlib import redirect_stdout

from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import AveragedModel

# Typing imports
from yacs.config import CfgNode


def dump_yacs_as_yaml(cfg: CfgNode, dump_path: str):
    """Dumps the YACS CfgNode as a YAML file. Useful for reproducibility, as new experiments can inherit the exact settings dumped to the local file.

    Args:
        cfg (CfgNode): YACS config object to dump.
        dump_path (str): Path to dump the YACS config. Generally should point to a `.yml` path in the experiment workdir.
    """
    with open(dump_path, "w") as f:
        with redirect_stdout(f):
            print(cfg.dump())


def save_model_checkpoint(
    save_fp: str,
    iter_step: int,
    model: nn.Module,
    model_ema: AveragedModel,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    best_metric: float,
):
    """Model checkpointing function.

    Args:
        save_fp (str): File path to save checkpoint to. Generally points to a `.pth` file.
        iter_step (int): The iteration step.
        model (nn.Module): The model. Will be used to save the model's state_dict.
        model_ema (AveragedModel): Exponential Moving Average (EMA) model. Will be skipped if not provided.
        optimizer (Optimizer): Optimizer. Used to save the state_dict for recovery or resume purposes.
        lr_scheduler (LRScheduler): Learning rate scheduler. Used to save the state_dict for recovery or resume purposes.
        best_metric (float): Best monitored metric. Used to resume model selection behaviour during recovery or resume.
    """
    if model_ema is None:
        torch.save(
            obj={
                "iter": iter_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "best_metric": best_metric,
            },
            f=save_fp,
        )
    else:
        torch.save(
            obj={
                "iter": iter_step,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": model_ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "best_metric": best_metric,
            },
            f=save_fp,
        )


def csv_logger(filepath: str, *args, reset: bool = False):
    """Rudimentary CSV logger to log experiments.

    An extremely basic CSV logger. Writes `*args` as new lines to the CSV at `filepath`.

    For example using `csv_logger("foo.csv", "bar", "baz")` will create `foo.csv` and write "bar,baz" to the file.

    Args:
        filepath (str): Path to the CSV file to write to.
        reset (bool, optional): Whether to create a new CSV file. Defaults to False.
    """
    if reset or not os.path.exists(filepath):
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(args)
    else:
        with open(filepath, "a+", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(args)


class CSVLogger(object):
    def __init__(self, filepath: str):
        """A more involved CSV writer that writes a dictionary instead of a list. This allows the headers to be maintained.

        Generally used for metric monitoring.

        Not technically safe as there is nothing enforcing key-safety or ordering.

        Args:
            filepath (str): Path to the CSV file to write to.
        """
        self.filepath = filepath

    def write(self, metric_dict: dict):
        """Write a metric dictionary to the CSV file. Will initialize headers if the CSV is freshly created.

        Args:
            metric_dict (dict): Metric dictionary to write to the file. Each key will correspond to a header, and the values will be written as a new line.
        """
        # If the CSV does not exist yet, create it while initializing the headers.
        if not os.path.exists(self.filepath):
            csv_logger(self.filepath, *list(metric_dict.keys()))
        # Write the metrics to the CSV file.
        csv_logger(self.filepath, *list(metric_dict.values()))
