import argparse
import os
import torch
import distutils

DEFAULT_CHANNELS = {
    "data": "data"
}


def strtobool(v):
    return bool(distutils.util.strtobool(v))


def bool_argument(
    parser: argparse.ArgumentParser,
    *args,
    **kwargs
):
    parser.add_argument(*args, **kwargs, type=strtobool, const=True, nargs="?")


def train_args_sagemaker(parser: argparse.ArgumentParser):
    bool_argument(
        parser, '--sagemaker-run'
    )
    pass

def train_args_export_model(parser: argparse.ArgumentParser,
                            model_dir='model',
                            export_model=True
                            ):
    model_dir = os.environ.get('SM_MODEL_DIR', model_dir)
    parser.add_argument(
        '--model-dir', type=str,
        default=model_dir,
        help='directory to save final model (default: {})'.format(model_dir))
    bool_argument(
        parser,
        '--export-model',
        default=export_model,
        help='export final model (boolean, default: {})'.format(export_model))


def train_args_hyperparameters(
    parser: argparse.ArgumentParser,
    batch_size=64,
    eval_batch_size=64,
    max_epochs=100,
    learning_rate=0.0003
):
    parser.add_argument("--workers", type=int, default=2,
                        help="number of data loading workers (default: 2)")
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',
                        help='input batch size for training (default: {})'.format(batch_size))
    parser.add_argument('--test-batch-size', type=int, default=eval_batch_size, metavar='N',
                        help='input batch size for testing (default: {})'.format(eval_batch_size))
    parser.add_argument('--max-epochs', type=int, default=max_epochs, metavar='N',
                        help='number of epochs to train (default: {})'.format(max_epochs))
    parser.add_argument('--learning-rate', type=float, default=learning_rate, metavar='LR',
                        help='learning rate (default: {})'.format(learning_rate))
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')


def train_args_worker(parser: argparse.ArgumentParser,
                      output_dir='output',
                      channels=None,
                      device=None):
    if channels is None:
        channels = DEFAULT_CHANNELS
    if device is None:
        device = "cpu" if not torch.cuda.is_available() else "cuda"
    output_dir = os.environ.get('SM_OUTPUT_DIR', output_dir)
    parser.add_argument("--device", type=str, default=device,
                        help="device to use (default: {})".format(device))
    parser.add_argument('--output-dir', type=str,
                        default=output_dir,
                        help='directory for checkpoints, logs, images, or other output files (default: "{}")'.format(output_dir))
    for channel, default in channels.items():
        key = 'SM_CHANNEL_{}'.format(channel.upper())
        if key in os.environ:
            default = os.environ[key]
        else:
            default = os.path.abspath(os.path.join(__file__, default))
        parser.add_argument('--{}'.format(channel), type=str,  default=default,
                            help="input directory for [{}] channel".format(channel))


def train_args_standard(parser: argparse.ArgumentParser,
                        output_dir='output',
                        batch_size=64,
                        eval_batch_size=64,
                        max_epochs=100,
                        learning_rate=0.0003,
                        channels=None,
                        device=None):
    train_args_hyperparameters(
        parser=parser,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate
    )
    train_args_worker(
        parser=parser,
        output_dir=output_dir,
        channels=channels,
        device=device
    )

    """
    parser.add_argument('--hosts', type=list,
                        default=json.loads(os.environ.get('SM_HOSTS', '[]')))
    parser.add_argument('--current-host', type=str,
                        default=os.environ.get('SM_CURRENT_HOST', ''))
    parser.add_argument('--num-gpus', type=int,
                        default=os.environ.get('SM_NUM_GPUS', 1))
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    train_args_standard(
        parser=parser,
        output_dir='output')
    train_args_export_model(
        parser=parser,
        model_dir='model'
    )
    parser.parse_args(args=['--help'])
