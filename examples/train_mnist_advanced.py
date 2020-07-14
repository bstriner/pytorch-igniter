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

from pytorch_igniter import train, get_value, RunSpec

# Define a model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=10,
                      kernel_size=3, stride=1),
        )

    def forward(self, input):
        return torch.mean(self.model(input), dim=(2, 3))


def train_mnist_classifier(dataroot, batch_size, workers, device,max_epochs, **train_args):

    # Create data loaders
    train_loader = data.DataLoader(dset.MNIST(root=dataroot, download=True, train=True,
                                              transform=transforms.ToTensor()), batch_size=batch_size,
                                   shuffle=True, num_workers=workers, drop_last=False)
    eval_loader = data.DataLoader(dset.MNIST(root=dataroot, download=True, train=False,
                                             transform=transforms.ToTensor()), batch_size=batch_size,
                                  shuffle=True, num_workers=workers, drop_last=False)

    # Create model, optimizer, and criteria
    model = Model().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    criteria = nn.CrossEntropyLoss(reduction='none')

    # Single step of training
    def train_step(engine, batch):
        # Put model into correct mode
        model.train()
        model.zero_grad()
        # Move batch to device
        pixels, labels = batch
        pixels, labels = pixels.to(device), labels.to(device)
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
            max_epochs=max_epochs
        ),
        eval_spec=RunSpec(
            step=eval_step,
            loader=eval_loader,
            metrics=metrics
        ),
        **train_args
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", help="path to dataset", default='data')
    parser.add_argument("--workers", type=int, default=2,
                        help="number of data loading workers")
    parser.add_argument("--batch-size", type=int,
                        default=32, help="input batch size")
    parser.add_argument("--max-epochs", type=int, default=25,
                        help="number of epochs to train for")
    parser.add_argument("--learning-rate", type=float, default=0.0003,
                        help="learning rate")
    parser.add_argument("--no-cuda", action="store_true", help="disables cuda")
    parser.add_argument("--output-dir", default='output/mnist/advanced',
                        help="directory to output images and model checkpoints")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = "cpu" if (not torch.cuda.is_available()
                       or args.no_cuda) else "cuda:0"
    train_mnist_classifier(
        dataroot=args.dataroot,
        batch_size=args.batch_size,
        workers=args.workers,
        output_dir=args.output_dir,
        device=device,
        max_epochs=args.max_epochs
    )


if __name__ == "__main__":
    main()
