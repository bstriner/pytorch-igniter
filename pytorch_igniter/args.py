import argparse
import os
import torch
import distutils
from aws_sagemaker_remote.args import bool_argument

TRAIN_KWARGS = [
    'n_saved',
    'output_dir',
    'model_dir',
    'checkpoint_dir',
    'mlflow_enable',
    'mlflow_tracking_uri',
    'device',
    'max_epochs',
    'mlflow_experiment_name',
    'mlflow_run_name',
    'is_sagemaker',
    'sagemaker_job_name',
    'mlflow_tracking_username',
    'mlflow_tracking_password',
    'mlflow_tracking_secret_name',
    'mlflow_tracking_secret_region',
    'mlflow_tracking_secret_profile'
]


def train_kwargs(args):
    return {k: getattr(args, k) for k in TRAIN_KWARGS if hasattr(args, k)}


def train_args(
    parser: argparse.ArgumentParser,
    batch_size=64,
    eval_batch_size=64,
    max_epochs=100,
    learning_rate=0.0003,
    device=None,
    mlflow_enable=True,
    mlflow_tracking_uri=None,
    mlflow_tracking_password=None,
    mlflow_tracking_username=None,
    mlflow_tracking_secret_name=None,
    mlflow_tracking_secret_region=None,
    mlflow_tracking_secret_profile=None,
    mlflow_experiment_name='default',
    mlflow_run_name=None,
    arguments='all',
    n_saved=5
):
    r"""
    Configure arguments for training

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser to configure.
    batch_size : int
        Batch size.
    arguments : list
        List of arguments to include (e.g. ``['batch_size``]) or the string  ``"all"`` to include all arguments.
        (default: "all")
    sagemaker_args : dict
        Keyword arguments to ``aws-sagemaker-remote.training.main.sagemaker_training_main``.
        See `aws_sagemaker_remote <https://aws-sagemaker-remote.readthedocs.io/en/latest/>`_.
    """

    if arguments == 'all' or 'workers' in arguments:
        parser.add_argument("--workers", type=int, default=2,
                            help="number of data loading workers (default: 2)")
    if arguments == 'all' or 'batch_size' in arguments:
        parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',
                            help='input batch size for training (default: {})'.format(batch_size))
    if arguments == 'all' or 'test_batch_size' in arguments:
        parser.add_argument('--test-batch-size', type=int, default=eval_batch_size, metavar='N',
                            help='input batch size for testing (default: {})'.format(eval_batch_size))
    if arguments == 'all' or 'max_epochs' in arguments:
        parser.add_argument('--max-epochs', type=int, default=max_epochs, metavar='N',
                            help='number of epochs to train (default: {})'.format(max_epochs))
    if arguments == 'all' or 'learning_rate' in arguments:
        parser.add_argument('--learning-rate', type=float, default=learning_rate, metavar='LR',
                            help='learning rate (default: {})'.format(learning_rate))
    if arguments == 'all' or 'seed' in arguments:
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
    if arguments == 'all' or 'mlflow_enable' in arguments:
        bool_argument(parser, '--mlflow-enable', default=mlflow_enable,
                      help='Enable logging to MLflow (default: {})'.format(mlflow_enable))
    if arguments == 'all' or 'mlflow_experiment_name' in arguments:
        parser.add_argument('--mlflow-experiment-name', default=mlflow_experiment_name,
                            help='Experiment name in MLflow (default: {})'.format(mlflow_experiment_name))
    if arguments == 'all' or 'mlflow_run_name' in arguments:
        parser.add_argument('--mlflow-run-name', default=mlflow_run_name,
                            help='Run name in MLflow (default: {})'.format(mlflow_run_name))
    if arguments == 'all' or 'mlflow_tracking_uri' in arguments:
        parser.add_argument('--mlflow-tracking-uri', default=mlflow_tracking_uri,
                            help='URI of MLflow tracking server (default: ``{}``)'.format(mlflow_tracking_uri))
    if arguments == 'all' or 'mlflow_tracking_username' in arguments:
        parser.add_argument('--mlflow-tracking-username', default=mlflow_tracking_username,type=str,
                            help='Username for MLflow tracking server (default: ``{}``)'.format(mlflow_tracking_username))
    if arguments == 'all' or 'mlflow_tracking_password' in arguments:
        parser.add_argument('--mlflow-tracking-password', default=mlflow_tracking_password,type=str,
                            help='Password for MLflow tracking server (default: ``{}``)'.format(mlflow_tracking_password))
    if arguments == 'all' or 'mlflow_tracking_secret_name' in arguments:
        parser.add_argument('--mlflow-tracking-secret-name', default=mlflow_tracking_secret_name,type=str,
                            help='Secret for accessing MLflow (default: ``{}``)'.format(mlflow_tracking_secret_name))
    if arguments == 'all' or 'mlflow_tracking_secret_profile' in arguments:
        parser.add_argument('--mlflow-tracking-secret-profile', default=mlflow_tracking_secret_profile,type=str,
                            help='Profile for accessing secret for accessing MLflow (default: ``{}``)'.format(mlflow_tracking_secret_profile))
    if arguments == 'all' or 'mlflow_tracking_secret_region' in arguments:
        parser.add_argument('--mlflow-tracking-secret-region', default=mlflow_tracking_secret_region,type=str,
                            help='Region for accessing secret for accessing MLflow (default: ``{}``)'.format(mlflow_tracking_secret_region))
    if arguments == 'all' or 'n_saved' in arguments:
        parser.add_argument('--n-saved', default=n_saved, type=int,
                            help='Number of checkpoints to keep (default: ``{}``)'.format(n_saved))
    if arguments == 'all' or 'device' in arguments:
        #if device is None:
        #    device = "cpu" if not torch.cuda.is_available() else "cuda"
        parser.add_argument("--device", type=str, default=device,
                            help="device to use (default: {})".format(device))

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


def parser_for_docs():
    parser = argparse.ArgumentParser()
    train_args(parser=parser)
    from aws_sagemaker_remote.training.args import sagemaker_training_args
    sagemaker_training_args(
        parser=parser,
        script='script.py'
    )
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    train_args(
        parser=parser
    )
    parser.parse_args(args=['--help'])
