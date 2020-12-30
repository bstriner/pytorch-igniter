from pytorch_igniter.args import train_args
import argparse


def test_train_args_default():
    parser = argparse.ArgumentParser()
    train_args(parser=parser, max_epochs=32)
    args = parser.parse_args()
    assert args.max_epochs == 32

