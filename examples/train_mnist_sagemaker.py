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
from pytorch_igniter.args import train_args_standard


def train_mnist_classifier(
    args
):

    # Create data loaders
    train_loader = data.DataLoader(MNIST(root=args.data, download=True, train=True,
                                              transform=transforms.ToTensor()), batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.workers, drop_last=False)
    eval_loader = data.DataLoader(MNIST(root=args.data, download=True, train=False,
                                             transform=transforms.ToTensor()), batch_size=args.batch_size,
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
        # Move batch to device
        pixels, labels = batch
        pixels, labels = pixels.to(args.device), labels.to(args.device)
        # Run model
        logits = model(pixels)
        # Run loss
        loss = criteria(input=logits, target=labels)
        # Calculate accuracy
        accuracy = torch.eq(torch.argmax(logits, dim=-1), labels).float()
        # Results must be scalar for RunningAverage
        loss = torch.mean(loss)
        accuracy = torch.mean(accuracy)
        # Train model
        loss.backward()
        optimizer.step()
        return {
            "loss": loss,
            "accuracy": accuracy
        }

    # Single step of evaluation
    def eval_step(engine, batch):
        # Put model into correct mode
        model.eval()
        # Move batch to device
        pixels, labels = batch
        pixels, labels = pixels.to(args.device), labels.to(args.device)
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
    to_save = {
        "model": model,
        "optimizer": optimizer
    }

    train(
        to_save=to_save,
        train_spec=RunSpec(
            step=train_step,
            loader=train_loader,
            metrics=metrics,
            max_epochs=args.max_epochs
        ),
        eval_spec=RunSpec(
            step=eval_step,
            loader=eval_loader,
            metrics=metrics
        ),
        #parameters=parameters,
        output_dir=args.output_dir
    )


def parse_args():
    parser = argparse.ArgumentParser()
    train_args_standard(
        parser=parser,
        output_dir='output/mnist/advanced/output',
        max_epochs=10,
        channels={
            'data': os.path.abspath(os.path.join(__file__,'../data'))
        }
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_mnist_classifier(args)


if __name__ == "__main__":
    main()
