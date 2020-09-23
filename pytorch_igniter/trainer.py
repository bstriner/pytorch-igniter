import argparse
import contextlib
import os
import random
import warnings
import numpy as np
from ignite.contrib.handlers.mlflow_logger import MLflowLogger
import mlflow
from .mlflow_ctx import mlflow_ctx, get_mlflow_logger
import re
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.utils.data as data

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from pytorch_igniter.metrics import SafeAverage

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import backward
import yaml
from .spec import RunSpec
from .engine import build_engine
from .util import handle_exception, get_last_checkpoint, get_metrics, capture_signals

LOADED = "Loaded {}, epoch {}, iteration {}"
COMPLETE = "Training complete"


def train(
    to_save,
    train_spec: RunSpec,
    eval_spec: RunSpec,
    eval_event=Events.EPOCH_COMPLETED,
    save_event=Events.EPOCH_COMPLETED,
    n_saved=10,
    output_dir=None,
    mlflow_enable=True,
    mlflow_tracking_uri=None,
    mlflow_experiment_name=None,
    mlflow_run_name=None,
    model_dir=None,
    parameters=None,
    device=None,
    max_epochs=None
):
    """
    Train a model
    """
    if max_epochs:
        train_spec.max_epochs = max_epochs
    if mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    ctx, output_dir = mlflow_ctx(
        output_dir=output_dir, mlflow_enable=mlflow_enable,
        experiment_name=mlflow_experiment_name, run_name=mlflow_run_name,
        parameters=parameters)
    os.makedirs(output_dir, exist_ok=True)
    with ctx:
        mlflow_logger = get_mlflow_logger(
            output_dir=output_dir,
            mlflow_enable=mlflow_enable
        )
        # Create trainer
        trainer = build_engine(
            spec=train_spec,
            output_dir=output_dir,
            mlflow_logger=mlflow_logger,
            tag='train',
            device=device
        )
        to_save = {'trainer': trainer, **to_save}

        # Saver
        checkpoint_handler = ModelCheckpoint(
            output_dir, filename_prefix="", n_saved=n_saved, require_empty=False)
        trainer.add_event_handler(
            event_name=save_event,
            handler=checkpoint_handler,
            to_save=to_save
        )

        # Optional evaluation
        if eval_spec is not None:
            assert eval_event is not None
            if not isinstance(eval_spec, dict):
                eval_spec = {
                    'eval': eval_spec
                }
            # Build evaluators
            evaluators = [
                (
                    build_engine(
                        spec=spec,
                        output_dir=output_dir,
                        mlflow_logger=mlflow_logger,
                        tag=tag,
                        trainer=trainer,
                        metric_cls=SafeAverage,
                        is_training=False,
                        device=device
                    ),
                    spec
                )
                for tag, spec in eval_spec.items()
            ]
            # Add evaluation hook to trainer

            def evaluation(engine):
                for evaluator, spec in evaluators:
                    evaluator.run(
                        spec.loader,
                        max_epochs=spec.max_epochs,
                        epoch_length=spec.epoch_length)
            trainer.add_event_handler(
                event_name=eval_event,
                handler=evaluation)

        # Handle ctrl-C or other exceptions
        def exception_callback(engine):
            # Save on exit
            if engine.state.iteration and engine.state.iteration > 0:
                _, last_iteration = get_last_checkpoint(
                    checkpoint_handler=checkpoint_handler)
                if last_iteration is None or last_iteration < engine.state.iteration:
                    checkpoint_handler(engine=engine, to_save=to_save)
        trainer.add_event_handler(
            event_name=Events.EXCEPTION_RAISED,
            handler=handle_exception,
            callback=exception_callback
        )

        # Get last checkpoint
        checkpoint_file, _ = get_last_checkpoint(checkpoint_handler)
        with capture_signals(
                callback=checkpoint_handler,
                engine=trainer,
                to_save=to_save):
            if checkpoint_file:
                # Load checkpoint
                checkpoint_data = torch.load(checkpoint_file)
                for key, value in to_save.items():
                    value.load_state_dict(checkpoint_data[key])
                tqdm.write(LOADED.format(
                    checkpoint_file, trainer.state.epoch, trainer.state.iteration))
                if Engine._is_done(trainer.state):
                    # Training complete
                    tqdm.write(COMPLETE)
                else:
                    # Continue training
                    trainer.run(train_spec.loader)
            else:
                # Start training
                trainer.run(
                    train_spec.loader,
                    max_epochs=train_spec.max_epochs,
                    epoch_length=train_spec.epoch_length)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        torch.save(
            {k: v.state_dict() for k, v in to_save.items()},
            os.path.join(model_dir, 'model.pt')
        )
    return get_metrics(engine=trainer)
