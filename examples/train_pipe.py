import argparse
import os
import pprint
from torch import nn
from torch.utils import data
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from aws_sagemaker_remote.training import sagemaker_training_main
import aws_sagemaker_remote
from aws_sagemaker_remote.args import PathArgument
import stat
import pathlib
import glob
from torch.utils.data import DataLoader
from scipy.io import wavfile
from io import BytesIO
import pytorch_igniter
from aws_sagemaker_remote.util.pipes import chunk_iterable
from sagemaker.amazon.record_pb2 import Record
from sagemaker.amazon.common import read_recordio
from pytorch_igniter.pipes import PipeDataset
import time


class MyPipeDataset(PipeDataset):
    def __init__(self, paths, buffering=0):
        print("Creating MyPipeDataset")
        super(MyPipeDataset, self).__init__(
            paths, size=2, buffering=buffering
        )

    def __iter__(self):
        for label, data in super(MyPipeDataset, self).__iter__():
            label = label.decode('utf-8')
            label = int(label)
            data = wavfile.read(BytesIO(data))
            yield label, data


def read_pipe(pipe):
    ds = MyPipeDataset(pipe, buffering=1024*1024*10)
    dl = DataLoader(
        ds,
        batch_size=None,
        num_workers=1,
        persistent_workers=True
    )
    for i in range(5):
        print("Epoch {}".format(i))
        it = 0
        for label, data in dl:
            # with open(pipe+"_{}".format(i), 'rb') as f:
            #    print("opened pipe {}".format(i))
            #    for label, f1, f2 in chunk_iterable(read_recordio(f), 3):
            #label = label.decode('utf-8')
            #fs, aud = wavfile.read(BytesIO(data))
            fs, aud = data
            print("{} label: {}".format(it, label))
            print("{} audio: {}, {}".format(it, fs, aud.shape))
            it += 1


def main(args):
    # Main function runs locally or remotely
    print("Test folder: {}".format(args.test_pipe))
    if isinstance(args.test_pipe, dict):
        for k, v in args.test_pipe.items():
            print("Pipe dict entry {}->{}".format(k, v))

            # print("Glob: {}".format(list(glob.glob(os.path.join(
            #    os.path.dirname(v), "**", "*"), recursive=True))))
            read_pipe(v)
    elif isinstance(args.test_pipe, str):
        print("Pipe {}".format(args.test_pipe))
        read_pipe(args.test_pipe)
    else:
        raise ValueError("Input should be string or dictionary")

    """
    show_path(args.test_pipe)
    print("Test S3 file pipe manifest: {}-manifest".format(args.test_pipe))
    show_path("{}-manifest".format(args.test_pipe))
    print("Input path: {}".format(os.path.dirname(args.test_pipe)))
    show_path(os.path.dirname(args.test_pipe))
    print("Test S3 file pipe 0: {}_0".format(args.test_pipe))
    read_pipe(args.test_pipe)
    """


if __name__ == '__main__':
    sagemaker_training_main(
        main=main,  # main function for local execution
        inputs={
            'test_pipe': PathArgument(
                'head.json',
                #'s3://sagemaker-us-east-1-683880991063/voxceleb-manifest-train-n-1-2020-10-27-07-15-40-631/output/output/train-manifest.json',
                mode='AugmentedManifestFile',
                attributes=['spid', 'file-ref']
            )
        },
        dependencies={
            # Add a module to SageMaker
            # module name: module path
            'aws_sagemaker_remote': aws_sagemaker_remote,
            'pytorch_igniter': pytorch_igniter
        },
        #configuration_command='pip3 install --upgrade sagemaker sagemaker-experiments',
        # Name the job
        base_job_name='demo-training-pipe',
        volume_size=20
    )

"""
split-lines --input demo/test_folder/manifest-speakers.json --output output/manifest-speakers --splits 2 --size 2
aws s3 sync demo/test_folder s3://sagemaker-us-east-1-683880991063/test_folder
python examples\train_pipe.py --sagemaker-run yes --sagemaker-training-instance ml.m5.large
"""
