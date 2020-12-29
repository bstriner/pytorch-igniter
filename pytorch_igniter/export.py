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
import pytorch_igniter.copyxattr_patch


def python_ignore(path, names):
    return [
        name for name in names
        if re.match("__pycache__|\\.git", name)
    ]


def export_code(model_dir, inference_args, inference_spec):
    os.makedirs(model_dir, exist_ok=True)
    if inference_args:
        with open(os.path.join(model_dir, 'args.json'), 'w') as f:
            json.dump(vars(inference_args), f)
    if inference_spec:
        # with open(os.path.join(model_dir,'config.json'),'w') as f:
        #    json.dump({
        #        "inferencer_module": inference_spec.inferencer.__class__.__module__,
        #        "inferencer_class": inference_spec.r.__class__.__name__,
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
                # dirs_exist_ok=True,
                ignore=python_ignore
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


def export_model(model_dir, model):
    os.makedirs(model_dir, exist_ok=True)
    torch.save(
        # {k: v.state_dict() for k, v in to_save.items()},
        {"model": model.state_dict()},
        os.path.join(model_dir, 'model.pt')
    )


def export_all(model_dir, model, inference_args, inference_spec):
    export_code(
        model_dir=model_dir,
        inference_args=inference_args,
        inference_spec=inference_spec)
    export_model(
        model_dir=model_dir,
        model=model)
