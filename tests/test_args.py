from pytorch_igniter.args import train_args
import argparse


def test_standard_args():
    parser = argparse.ArgumentParser()
    train_args(parser=parser, batch_size=123)
    args = parser.parse_args()
    assert args.batch_size == 123
