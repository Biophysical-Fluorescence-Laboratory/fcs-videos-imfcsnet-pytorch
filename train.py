import os
import argparse
import datetime

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from imfcs_pytorch.config.initialization import get_default_cfg, validate_cfg
from imfcs_pytorch.utils.tracking import (
    dump_yacs_as_yaml,
    save_model_checkpoint,
    CSVLogger,
)
from imfcs_pytorch.builders.optimizers import build_optimizer
from imfcs_pytorch.builders.schedulers import build_lr_scheduler

# Model import
from imfcs_pytorch.builders.model import build_model
from imfcs_pytorch.builders.transforms import (
    build_train_transforms,
    build_target_transforms,
)
from imfcs_pytorch.builders.dataset import build_simulator

# Typing-only imports, only used for typehints
from yacs.config import CfgNode
from typing import List


def get_parser() -> argparse.ArgumentParser:
    """Creates the parser to initialize an experiment.

    This should follow the YACS convention of 'There is only one way to configure the same thing.' Preferably, no additional CLI arguments should be added here. Instead, add them to the YACS configuration file, such that they can be overriden using the --config-overrides option.

    Returns:
        argparse.ArgumentParser: CLI Argument Parser.
    """
    parser = argparse.ArgumentParser(description="Image classification training.")
    parser.add_argument(
        "--cfg",
        required=True,
        metavar="FILE",
        help="Path to YACS config file in .yaml format.",
    )
    parser.add_argument(
        "--overrides",
        metavar="STRING",
        default=[],
        type=str,
        help=(
            "Modify experimental configuration from the command line. See https://github.com/rbgirshick/yacs#command-line-overrides for details. Inputs should be comma-separated: 'python train.py --config-overrides EXPERIMENT.NAME modified_exp MODEL.NAME timm_resnet_18'."
        ),
        nargs=argparse.ONE_OR_MORE,
    )
    parser.add_argument(
        "--ckpt",
        required=False,
        default=None,
        metavar="FILE",
        help="Path to an existing checkpoint, will trigger the resume function.",
    )

    return parser


def setup(config_path: str, cli_overrides: List[str]) -> CfgNode:
    """Initialize the experimental configuration, and return an immutable configuration as per the YACS convention.

    Args:
        config_path (str): Path to the YACS config file.
        cli_overrides (List[str]): CLI overrides for experimental parameters. Should be a list of

    Returns:
        CfgNode: _description_
    """
    cfg = get_default_cfg()

    # Merge overrides from the configuration file and command-line overrides.
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(cli_overrides)

    # Validate the config file settings through assertions defined in classification/config/initialization.py
    validate_cfg(cfg)

    # If no seed is specified, generate a random seed as defined by the NumPy docs
    # This is here to ensure reproducibility, as the randomly generate seed will still be dumped into the workdir.
    if cfg.EXPERIMENT.SEED is None:
        # https://numpy.org/doc/stable/reference/random/index.html#random-quick-start
        import secrets

        cfg.EXPERIMENT.SEED = secrets.randbits(32)

    # Freeze the config file to make it immutable.
    cfg.freeze()

    return cfg


def main(cfg: CfgNode, resume_ckpt_path: str = None):
    """Conducts an experiment based on the input experimental config.

    Args:
        cfg (CfgNode): Experimental configuration in the YACS format.
    """
    # Perform seeding.
    torch.manual_seed(cfg.EXPERIMENT.SEED)
    np.random.seed(cfg.EXPERIMENT.SEED)

    # If active, turn on CUDNN benchmarking.
    if cfg.BACKEND.CUDNN_BENCHMARK:
        cudnn.benchmark = True

    # Extract the CUDA device that will be used.
    # TODO: Try to make this work with device="cpu", tentatively difficuly because the device also links to the CUDA compile function.
    device = cfg.EXPERIMENT.DEVICE

    # Create the transform chain
    train_transforms = build_train_transforms(
        cfg=cfg,
        device=device,
        disable_flag=cfg.TRANSFORMS.TRAIN.DISABLE,
    )

    # Create target transforms if applicable.
    target_transforms = build_target_transforms(cfg.TASK.REGRESSION.TARGET_TRANSFORM)

    # Build the datasets.
    # TODO: Modularize this, might include non-simulation-based datasets in the future.
    # First, build the simulator
    simulator = build_simulator(cfg=cfg, device=device)
    simulator.start_simulation()  # Start the simulations.
    dataset_train = simulator.get_dataset(
        transforms=train_transforms,
    )

    # Building the dataloaders.
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=cfg.DATALOADER.PER_STEP_BATCH_SIZE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=cfg.BACKEND.DATALOADER_PIN_MEMORY,
    )

    # Build model
    model = build_model(
        cfg=cfg, output_list=cfg.TASK[cfg.EXPERIMENT.TASK]["TARGETS"], device=device
    )

    # Build loss criterion
    criterion = nn.MSELoss()

    # As per most modern publications, the learning rate is scaled based on the batch size.
    # absolute_lr = base_lr * total_batch_size / 256
    learning_rate = cfg.OPTIMIZER.BASE_LEARNING_RATE
    # if cfg.BACKEND.SCALE_LEARNING_RATE:
    #     print("Scaling learning rate based on batch size.")
    #     print(f"Apparent batch size = {total_batch_size}")
    #     learning_rate = learning_rate * total_batch_size / 256
    #     print(f"Scaled learning rate is {learning_rate}")

    # Build optimizer
    optimizer = build_optimizer(
        param_groups=model.parameters(),
        optimizer_type=cfg.OPTIMIZER.TYPE,
        learning_rate=learning_rate,
        weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        momentum=cfg.OPTIMIZER.MOMENTUM,
        betas=cfg.OPTIMIZER.BETAS,
        eps=cfg.OPTIMIZER.EPS,
    )

    # Build the learning rate scheduler
    lr_scheduler = build_lr_scheduler(
        optimizer=optimizer,
        lr_scheduler_type=cfg.LR_SCHEDULER.TYPE,
        total_steps=cfg.TRAINING.ITERATIONS,
        warmup_steps=cfg.LR_SCHEDULER.WARMUP_STEPS,
        min_lr=cfg.LR_SCHEDULER.MIN_LR,
        update_period=cfg.LR_SCHEDULER.UPDATE_PERIOD,
        milestones=cfg.LR_SCHEDULER.MILESTONES,
        lr_multiplicative_factor=cfg.LR_SCHEDULER.LR_MULTIPLICATIVE_FACTOR,
    )

    # Everything is good to go.
    # Before starting the training loop, initializing the workdir and logging tools.
    workdir_path = cfg.EXPERIMENT.WORKDIR
    checkpoints_folder = os.path.join(workdir_path, "checkpoints")
    logs_folder = os.path.join(workdir_path, "logs")
    os.makedirs(workdir_path, exist_ok=True)
    os.makedirs(checkpoints_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    # Create the interval checkpointing folder if requested.
    if cfg.EXPERIMENT.CHECKPOINTING.DO_INTERVAL_CHECKPOINTING:
        os.makedirs(os.path.join(checkpoints_folder, "intervals"), exist_ok=True)

    # Initialize loggers.
    iter_csv_path = os.path.join(logs_folder, "iter_logs.csv")
    iter_csv_logger = CSVLogger(iter_csv_path)

    if cfg.LOGGING.DO_TENSORBOARD_LOGGING:
        tensorboard_writer = SummaryWriter(os.path.join(logs_folder, "tensorboard"))

    # Make a dump of the config file for reproducibility.
    dump_yacs_as_yaml(cfg, os.path.join(workdir_path, "config.yaml"))

    # Training loop.
    # Set default loss to be high
    best_metric = 999999  # Training loss is used.
    model.train()
    for data_iter_step, (samples, targets) in enumerate(dataloader_train, start=0):
        samples = samples.unsqueeze(1)

        # Apply transformations for the targets if requested. (i.e. log, log10 etc.)
        for class_ind, target_transform in enumerate(target_transforms):
            targets[:, class_ind] = target_transform(targets[:, class_ind])

        targets = targets.float().to(device, non_blocking=True)

        output = model(samples)
        loss = criterion(output, targets)

        # Optional flag that allows the code to break if NaN loss is achieved.
        if cfg.BACKEND.BREAK_WHEN_LOSS_NAN and torch.isnan(loss):
            raise RuntimeError(f"Loss became NaN at step {data_iter_step}.")

        # Backpropagate.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss = loss.detach().cpu().item()

        # Log per-iteration
        print(
            f"{data_iter_step % cfg.TRAINING.ITERATIONS + 1}/{cfg.TRAINING.ITERATIONS}, train_loss: {training_loss:.4f}"
        )
        if data_iter_step % cfg.LOGGING.LOGGING_ITER_INTERVAL == 0:
            iter_step_dict = {
                "step": data_iter_step,
                "learning_rate": lr_scheduler.get_last_lr()[0],
                "training_loss_iter": training_loss,
                "time": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            }

            iter_csv_logger.write(iter_step_dict)
            iter_step_dict.pop("time")
            if cfg.LOGGING.DO_TENSORBOARD_LOGGING:
                # Pop the "iter_step" key, since Tensorboard logs this as the 'global_step' parameter.
                iter_step_dict.pop("step")

                for key, value in iter_step_dict.items():
                    tensorboard_writer.add_scalar(key, value, data_iter_step)

        # This case is triggered when there is no evaluation set. In this case, the training loss is used for model selection.
        if training_loss < best_metric:
            print(
                f"Best training loss improved from {best_metric:.4e} to {training_loss:.4e}"
            )
            best_metric = training_loss
            _save_model_path = os.path.join(checkpoints_folder, "best_model.ckpt")
            print(f"Saving new best model to {_save_model_path}")
            save_model_checkpoint(
                _save_model_path,
                data_iter_step,
                model,
                None,
                optimizer,
                lr_scheduler,
                best_metric,
            )

        # After every epoch, save a checkpoint
        save_model_checkpoint(
            os.path.join(checkpoints_folder, "last.ckpt"),
            data_iter_step,
            model,
            None,
            optimizer,
            lr_scheduler,
            best_metric,
        )

        # Save in intervals
        if cfg.EXPERIMENT.CHECKPOINTING.DO_INTERVAL_CHECKPOINTING:
            if (
                data_iter_step
            ) % cfg.EXPERIMENT.CHECKPOINTING.CHECKPOINTING_INTERVAL == 0:
                save_model_checkpoint(
                    os.path.join(
                        checkpoints_folder,
                        "intervals",
                        f"interval_{data_iter_step}.ckpt",
                    ),
                    data_iter_step,
                    model,
                    None,
                    optimizer,
                    lr_scheduler,
                    best_metric,
                )

        # After an epoch has passed, advance the lr_scheduler.
        lr_scheduler.step()
    simulator.stop_simulation()


if __name__ == "__main__":
    # Parse arguments from the CLI.
    args = get_parser().parse_args()
    config_file = args.cfg
    cli_overrides = args.overrides
    existing_ckpt = args.ckpt

    # Generates the experimental configuration node.
    cfg = setup(config_path=config_file, cli_overrides=cli_overrides)

    # Conduct the training loop.
    main(cfg, existing_ckpt)
