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
    'mlflow_tracking_secret_profile',
    'eval_event',
    'save_event',
    'eval_pbar',
    'train_pbar',
    'train_print_event',
    'eval_print_event',
    'train_log_event',
    'eval_log_event'
]


def train_kwargs(args):
    return {k: getattr(args, k) for k in TRAIN_KWARGS if hasattr(args, k)}


def fix_device(args):
    vargs = vars(args)
    if not vargs.get('device', None):
        vargs['device'] = "cpu" if not torch.cuda.is_available() else "cuda"
        print("Detected device=[{}]".format(vargs['device']))
    else:
        print("Selected device=[{}]".format(vargs['device']))
    args = argparse.Namespace(**vargs)
    return args


def mlflow_args(
        parser,
        mlflow_enable=True,
        mlflow_tracking_uri=None,
        mlflow_tracking_password=None,
        mlflow_tracking_username=None,
        mlflow_tracking_secret_name=None,
        mlflow_tracking_secret_region=None,
        mlflow_tracking_secret_profile=None,
        mlflow_experiment_name='default',
        mlflow_run_name=None):
    bool_argument(parser, '--mlflow-enable', default=mlflow_enable,
                  help='Enable logging to MLflow (default: {})'.format(mlflow_enable))
    parser.add_argument('--mlflow-experiment-name', default=mlflow_experiment_name,
                        help='Experiment name in MLflow (default: {})'.format(mlflow_experiment_name))
    parser.add_argument('--mlflow-run-name', default=mlflow_run_name,
                        help='Run name in MLflow (default: {})'.format(mlflow_run_name))
    parser.add_argument('--mlflow-tracking-uri', default=mlflow_tracking_uri,
                        help='URI of MLflow tracking server (default: ``{}``)'.format(mlflow_tracking_uri))
    parser.add_argument('--mlflow-tracking-username', default=mlflow_tracking_username, type=str,
                        help='Username for MLflow tracking server (default: ``{}``)'.format(mlflow_tracking_username))
    parser.add_argument('--mlflow-tracking-password', default=mlflow_tracking_password, type=str,
                        help='Password for MLflow tracking server (default: ``{}``)'.format(mlflow_tracking_password))
    parser.add_argument('--mlflow-tracking-secret-name', default=mlflow_tracking_secret_name, type=str,
                        help='Secret for accessing MLflow (default: ``{}``)'.format(mlflow_tracking_secret_name))
    parser.add_argument('--mlflow-tracking-secret-profile', default=mlflow_tracking_secret_profile, type=str,
                        help='Profile for accessing secret for accessing MLflow (default: ``{}``)'.format(mlflow_tracking_secret_profile))
    parser.add_argument('--mlflow-tracking-secret-region', default=mlflow_tracking_secret_region, type=str,
                        help='Region for accessing secret for accessing MLflow (default: ``{}``)'.format(mlflow_tracking_secret_region))


def train_and_eval_args(
    parser: argparse.ArgumentParser,
    eval_event='EPOCH_COMPLETED'
):
    parser.add_argument(
        '--eval-event', default=eval_event, help='Evaluation event'
    )
    parser.add_argument(
        '--eval-print-event', default=None, help='Evaluation print event'
    )
    parser.add_argument(
        '--eval-log-event', default=None, help='Evaluation log event'
    )


def train_args(parser, max_epochs=10, n_saved=10, save_event='EPOCH_COMPLETED'):
    parser.add_argument(
        '--max-epochs', type=int, default=max_epochs, metavar='N',
        help='number of epochs to train (default: {})'.format(max_epochs))
    parser.add_argument(
        '--n-saved', default=n_saved, type=int,
        help='Number of checkpoints to keep (default: ``{}``)'.format(n_saved))
    parser.add_argument(
        '--save-event', default=save_event, help='save event'
    )
    bool_argument(
        parser,
        '--train-pbar', default=True, help='Enable train progress bar'
    )
    parser.add_argument(
        '--train-print-event', default=None, help='training print event'
    )
    parser.add_argument(
        '--train-log-event', default=None, help='training log event'
    )
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                        help='random seed (default: 1)')


def eval_args(
    parser: argparse.ArgumentParser
):
    bool_argument(
        parser,
        '--eval-pbar', default=True, help='Enable eval progress bar'
    )


def model_args(parser, device=None):
    parser.add_argument("--device", type=str, default=device,
                        help="device to use (default: {})".format(device))

def add_model_args(parser,args):
    group = parser.add_argument_group(
            title='Model',
            description='Model arguments'
        )
    model_args(group)
    if args:
        args(group)
def add_train_args(parser, args):
    group = parser.add_argument_group(
            title='Training',
            description='Training arguments'
        )
    train_args(group)
    if args:
        args(group)
def add_eval_args(parser,args):
    group = parser.add_argument_group(
            title='Evaluation',
            description='Evaluation arguments'
        )
    eval_args(group)
    if args:
        args(group)
def add_tran_and_eval_args(parser):
    group = parser.add_argument_group(
            title='Train/Eval schedule',
            description='Arguments affecting scheduling'
        )
    train_and_eval_args(group)

def add_mlflow_args(parser):
    group = parser.add_argument_group(
        title='MLflow',
        description='MLflow arguments'
    )
    mlflow_args(group)

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
