import argparse
import contextlib
import os
import random
import warnings
import numpy as np
from ignite.contrib.handlers.mlflow_logger import MLflowLogger
import mlflow
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
from .util import handle_exception, get_last_checkpoint, get_metrics
RUN_FNAME = 'run.yaml'
LOADED = "Loaded {}, epoch {}, iteration {}"
COMPLETE = "Training complete"


def train(
    to_save,
    output_dir,
    train_spec: RunSpec,
    eval_spec: RunSpec,
    eval_event=Events.EPOCH_COMPLETED,
    save_event=Events.EPOCH_COMPLETED,
    n_saved=10,
    mlflow_enable=True,
    mlflow_tracking_uri=None,
    parameters=None
):
    if mlflow_enable:
        active_run = mlflow.active_run()
        run_id = None
        if 'MLFLOW_RUN_ID' in os.environ:
            print("Active MLflow run")
            run_id = os.environ['MLFLOW_RUN_ID']
            output_dir = os.path.join(output_dir, run_id)
            run_fname = os.path.join(output_dir, RUN_FNAME)
        else:
            run_fname = os.path.join(output_dir, RUN_FNAME)
            if os.path.exists(run_fname):
                print("Resume MLflow run")
                with open(run_fname) as f:
                    run_id = yaml.load(f, Loader=yaml.SafeLoader)[
                        'info']['run_id']
            else:
                print("New MLflow run")
                run_id = None
        ctx = mlflow.start_run(run_id=run_id)
    else:
        ctx = contextlib.nullcontext()

    os.makedirs(output_dir, exist_ok=True)

    with ctx:
        if mlflow_enable:
            mlflow_logger = MLflowLogger(tracking_uri=mlflow_tracking_uri)
            active_run = mlflow.active_run()
            if not os.path.exists(run_fname):
                if parameters is not None:
                    mlflow.log_params(parameters)
                active_run = mlflow.get_run(active_run.info.run_id)
                with open(run_fname, 'w') as f:
                    yaml.dump(active_run.to_dictionary(), f)
        else:
            mlflow_logger = None

        # Create trainer
        trainer = build_engine(
            spec=train_spec,
            output_dir=output_dir,
            mlflow_logger=mlflow_logger,
            tag='train'
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
                        metric_cls=SafeAverage
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
    return get_metrics(engine=trainer)
