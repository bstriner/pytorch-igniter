import os
from argparse import Namespace
import importlib
import json
import torch

"""
Dummy inference script echoes input as output
"""

print("Loaded inference.py: {}".format(__file__))


def model_fn(model_dir):
    """
    Load your model from model_dir
    """
    print("Running model_fn: {}".format(model_dir))
    # with open(os.path.join(model_dir, 'config.json'), 'r') as f:
    #    config = json.load(f)
    with open(os.path.join(model_dir, 'args.json'), 'r') as f:
        vargs = json.load(f)
    args = Namespace(**vargs)
    # inferencer_module = importlib.import_module(
    #    config['inferencer_module']
    # )
    #inferencer_class = getattr(inferencer_module, config['inferencer_class'])
    inferencer = inferencer_fn(args)
    model = inferencer.model
    loaded = torch.load(os.path.join(model_dir, 'model.pt'))
    model.load_state_dict(loaded['model'])
    return inferencer


def predict_fn(input_data, model):
    """
    Run your model
    """
    print("Running predict_fn")
    return model.inference(input_data)
