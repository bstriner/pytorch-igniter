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

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import backward
from pytorch_igniter.demo.mnist_model import MnistModel
from pytorch_igniter import train, get_value, RunSpec
from pytorch_igniter.evaluator import evaluate


def evaluate_mnist_classifier(output_dir, dataroot, batch_size, workers, device, max_epochs, learning_rate, **train_args):

    # Create data loader
    eval_loader = data.DataLoader(dset.MNIST(root=dataroot, download=True, train=False,
                                             transform=transforms.ToTensor()), batch_size=batch_size,
                                  shuffle=True, num_workers=workers, drop_last=False)

    # Create model and criteria
    model = MnistModel().to(device)
    criteria = nn.CrossEntropyLoss(reduction='none')

    # Single step of evaluation
    def eval_step(engine, batch):
        # Put model into correct mode
        model.eval()
        # Move batch to device
        pixels, labels = batch
        pixels, labels = pixels.to(device), labels.to(device)
        # Run model
        logits = model(pixels)
        # Run loss
        loss = criteria(input=logits, target=labels)
        # Calculate accuracy
        accuracy = torch.eq(torch.argmax(logits, dim=-1), labels).float()
        # Results must be shaped (n, 1) for Average to know the batch size
        loss = loss.view(-1, 1)
        accuracy = accuracy.view(-1, 1)
        return {
            "loss": loss,
            "accuracy": accuracy
        }

    # Metrics average the outputs of the step functions and are printed and saved to logs
    metrics = {
        'loss': 'loss',
        'accuracy': 'accuracy'
    }

    # Objects to save
    to_load = {
        "model": model
    }

    eval_spec = RunSpec(
        loader=eval_loader,
        step=eval_step,
        metrics=metrics,
        log_event=None,
        plot_event=None
    )

    return evaluate(
        eval_spec=eval_spec,
        output_dir=output_dir,
        to_load=to_load
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", help="path to dataset", default='data')
    parser.add_argument("--workers", type=int, default=2,
                        help="number of data loading workers")
    parser.add_argument("--batch-size", type=int,
                        default=32, help="input batch size")
    parser.add_argument("--max-epochs", type=int, default=10,
                        help="number of epochs to train for")
    parser.add_argument("--learning-rate", type=float, default=0.0003,
                        help="learning rate")
    parser.add_argument("--device", type=str,
                        default='cuda:0' if torch.cuda.is_available() else 'cpu', help="device to use")
    parser.add_argument("--output-dir", default='output/mnist/advanced',
                        help="directory to output images and model checkpoints")
    parser.add_argument("--mlflow-tracking-uri", type=str,
                        help="directory to output images and model checkpoints")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    evaluate_mnist_classifier(
        **vars(args)
    )


if __name__ == "__main__":
    main()
