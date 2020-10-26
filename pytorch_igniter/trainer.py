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
from aws_sagemaker_remote.modules import module_path
import shutil
from .events import event_argument
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.utils.data as data
import pytorch_igniter.inference.inference
import json
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
from .ssm import get_secret
import re
LOADED = "Loaded {}, epoch {}, iteration {}"
COMPLETE = "Training complete"

# have to monkey patch to work with WSL as workaround for https://bugs.python.org/issue38633
import errno, shutil
orig_copyxattr = shutil._copyxattr
def patched_copyxattr(src, dst, *, follow_symlinks=True):
	try:
		orig_copyxattr(src, dst, follow_symlinks=follow_symlinks)
	except OSError as ex:
		if ex.errno != errno.EACCES: raise
shutil._copyxattr = patched_copyxattr

def ignore(path, names):
    return [
        name for name in names
        if re.match("__pycache__|\\.git", name)
    ]


def train(
    to_save,
    model,
    train_spec: RunSpec,
    eval_spec: RunSpec = None,
    eval_event=Events.EPOCH_COMPLETED,
    save_event=Events.EPOCH_COMPLETED,
    n_saved=10,
    mlflow_enable=True,
    mlflow_tracking_uri=None,
    mlflow_tracking_username=None,
    mlflow_tracking_password=None,
    mlflow_tracking_secret_name=None,
    mlflow_tracking_secret_profile=None,
    mlflow_tracking_secret_region=None,
    mlflow_experiment_name=None,
    mlflow_run_name=None,
    model_dir='output',
    checkpoint_dir='output',
    output_dir='output',
    parameters=None,
    device=None,
    max_epochs=None,
    is_sagemaker=False,
    sagemaker_job_name=None,
    inference_spec=None,
    inference_args=None
):
    """
    Train a model
    """
    save_event = event_argument(save_event)
    eval_event = event_argument(eval_event)
    if max_epochs:
        train_spec.max_epochs = max_epochs
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    if mlflow_tracking_username:
        os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_tracking_username
    if mlflow_tracking_password:
        os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_tracking_password
    if mlflow_tracking_secret_name:
        secret = get_secret(
            profile_name=mlflow_tracking_secret_profile,
            secret_name=mlflow_tracking_secret_name,
            region_name=mlflow_tracking_secret_region)
        if not secret:
            raise ValueError("Could not get secret [{}]. Check secret name, region, and role permissions".format(
                mlflow_tracking_secret_name))
        uri = secret.get('uri', None)
        username = secret.get('username', None)
        password = secret.get('password', None)
        if uri:
            # print("Set uri from secret: [{}]".format(uri))
            mlflow.set_tracking_uri(uri)
        if username:
            # print("Set username from secret")
            os.environ['MLFLOW_TRACKING_USERNAME'] = username
        if password:
            # print("Set password from secret")
            os.environ['MLFLOW_TRACKING_PASSWORD'] = password
    if 'MLFLOW_RUN_ID' in os.environ:
        run_id = os.environ['MLFLOW_RUN_ID']
        # output_dir = os.path.join(output_dir, run_id)
        # model_dir = os.path.join(model_dir, run_id)
        # checkpoint_dir = os.path.join(checkpoint_dir, run_id)

    ctx = mlflow_ctx(
        output_dir=output_dir, checkpoint_dir=checkpoint_dir, mlflow_enable=mlflow_enable,
        experiment_name=mlflow_experiment_name, run_name=mlflow_run_name,
        parameters=parameters, is_sagemaker=is_sagemaker, sagemaker_job_name=sagemaker_job_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with ctx:
        mlflow_logger = get_mlflow_logger(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
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
            checkpoint_dir, filename_prefix="", n_saved=n_saved, require_empty=False)

        def safe_checkpoint_handler(engine, to_save):
            if engine.state.iteration and engine.state.iteration > 0:
                _, last_iteration = get_last_checkpoint(
                    checkpoint_handler=checkpoint_handler)
                if last_iteration is None or last_iteration < engine.state.iteration:
                    checkpoint_handler(engine=engine, to_save=to_save)

        trainer.add_event_handler(
            event_name=save_event,
            handler=safe_checkpoint_handler,
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
            safe_checkpoint_handler(engine=engine, to_save=to_save)
        trainer.add_event_handler(
            event_name=Events.EXCEPTION_RAISED,
            handler=handle_exception,
            callback=exception_callback
        )

        # Get last checkpoint
        checkpoint_file, _ = get_last_checkpoint(checkpoint_handler)
        with capture_signals():
            if checkpoint_file:
                # Load checkpoint
                checkpoint_data = torch.load(checkpoint_file)
                print(checkpoint_file)
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
        safe_checkpoint_handler(engine=trainer, to_save=to_save)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            # todo just export model
            torch.save(
                # {k: v.state_dict() for k, v in to_save.items()},
                {"model": model.state_dict()},
                os.path.join(model_dir, 'model.pt')
            )
            if inference_args:
                with open(os.path.join(model_dir, 'args.json'), 'w') as f:
                    json.dump(vars(inference_args), f)
            if inference_spec:
                # with open(os.path.join(model_dir,'config.json'),'w') as f:
                #    json.dump({
                #        "inferencer_module": inference_spec.inferencer.__class__.__module__,
                #        "inferencer_class": inference_spec.inferencer.__class__.__name__,
                #    }, f)
                code_dir = os.path.join(model_dir, 'code')
                os.makedirs(code_dir, exist_ok=True)
                for dep in inference_spec.dependencies:
                    dep = module_path(dep)
                    des = os.path.join(
                            code_dir, os.path.basename(dep)
                        )
                    if os.path.exists(des):
                        shutil.rmtree(des)
                    shutil.copytree(
                        dep,
                        des,
                        #dirs_exist_ok=True,
                        ignore=ignore
                    )
                if inference_spec.requirements:
                    shutil.copyfile(inference_spec.requirements, os.path.join(
                        code_dir, 'requirements.txt'
                    ))
                inference_py = os.path.join(code_dir, 'inference.py')
                shutil.copyfile(
                    pytorch_igniter.inference.inference.__file__,
                    inference_py
                )
                with open(inference_py, 'a') as f:
                    f.write("\n")
                    f.write("\n")
                    from types import FunctionType
                    if isinstance(inference_spec.input_fn, str):
                        f.write("{}\n".format(inference_spec.input_fn))
                    else:
                        f.write("from {} import {} as input_fn\n".format(
                            inference_spec.input_fn.__module__,
                            inference_spec.input_fn.__name__
                        ))
                    if isinstance(inference_spec.output_fn, str):
                        f.write("{}\n".format(inference_spec.output_fn))
                    else:
                        f.write("from {} import {} as output_fn\n".format(
                            inference_spec.output_fn.__module__,
                            inference_spec.output_fn.__name__
                        ))
                    if isinstance(inference_spec.inferencer, str):
                        f.write("{}\n".format(inference_spec.inferencer))
                    else:
                        f.write("from {} import {} as inferencer_fn\n".format(
                            inference_spec.inferencer.__module__,
                            inference_spec.inferencer.__name__
                        ))

        return get_metrics(engine=trainer)
