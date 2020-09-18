from pytorch_igniter.args import train_args_standard
import argparse


def test_standard_args():
    parser = argparse.ArgumentParser()
    train_args_standard(parser=parser, output_dir='testoutputdir')
    args = parser.parse_args()
    assert args.output_dir == 'testoutputdir'
