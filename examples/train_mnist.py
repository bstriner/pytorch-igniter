import argparse
import os
import random
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Average

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import backward
from pytorch_igniter.demo.mnist_model import MnistModel
from pytorch_igniter import train, get_value, RunSpec
from pytorch_igniter.args import train_kwargs
from pytorch_igniter.main import igniter_main


def main(
    args
):

    # Create data loaders
    train_loader = data.DataLoader(MNIST(root=args.data, download=True, train=True,
                                         transform=transforms.ToTensor()), batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.workers, drop_last=False)
    eval_loader = data.DataLoader(MNIST(root=args.data, download=True, train=False,
                                        transform=transforms.ToTensor()), batch_size=args.test_batch_size,
                                  shuffle=True, num_workers=args.workers, drop_last=False)

    # Create model, optimizer, and criteria
    model = MnistModel().to(args.device)
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.learning_rate)
    criteria = nn.CrossEntropyLoss(reduction='none')

    # Single step of training
    def train_step(engine, batch):
        # Put model into correct mode
        model.train()
        model.zero_grad()
        pixels, labels = batch
        # Run model
        logits = model(pixels)
        # Run loss
        loss = torch.mean(criteria(input=logits, target=labels))
        # Calculate accuracy
        accuracy = torch.mean(
            torch.eq(torch.argmax(logits, dim=-1), labels).float())
        # Train model
        loss.backward()
        optimizer.step()
        # Outputs should be scalars during training
        return {
            "loss": loss,
            "accuracy": accuracy
        }

    # Single step of evaluation
    def eval_step(engine, batch):
        # Put model into correct mode
        model.eval()
        pixels, labels = batch
        # Run model
        logits = model(pixels)
        # Run loss
        loss = criteria(input=logits, target=labels)
        # Calculate accuracy
        accuracy = torch.eq(torch.argmax(logits, dim=-1), labels).float()
        # Outputs should be shaped (n, 1) during evaluation
        return {
            "loss": loss.view(-1, 1),
            "accuracy": accuracy.view(-1, 1)
        }

    # Metrics average the outputs of the step functions and are printed and saved to logs
    metrics = {
        'loss': 'loss',
        'accuracy': 'accuracy'
    }

    # Objects to save
    to_save = {
        "model": model,
        "optimizer": optimizer
    }

    train(
        to_save=to_save,
        train_spec=RunSpec(
            step=train_step,
            loader=train_loader,
            metrics=metrics
        ),
        eval_spec=RunSpec(
            step=eval_step,
            loader=eval_loader,
            metrics=metrics
        ),
        **train_kwargs(args),
        parameters=vars(args)
    )


if __name__ == "__main__":
    igniter_main(
        main=main,
        training_args={
            'max_epochs': 5,  # change default epochs (optional)
            # experiment name in MLflow (optional)
            'mlflow_experiment_name': 'mnist-demo'
        },
        inputs={
            'data': os.path.abspath(os.path.join(__file__, '../../output/mnist/data'))
        },
        # change default output path (optional)
        output_dir='output/mnist/output',
        # change default model export path (optional)
        model_dir='output/mnist/model',
        # change default model checkpoint path (optional)
        checkpoint_dir='output/mnist/checkpoint',
        # experiment name in SageMaker (optional)
        experiment_name='mnist-demo',
        # argparse description (optional)
        description='Demo script for MNIST training'
    )

# python examples/train_mnist.py --sagemaker-run yes --sagemaker-spot-instances yes --mlflow-tracking-uri ...
