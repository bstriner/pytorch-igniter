import argparse

from aws_sagemaker_remote.training.args import sagemaker_training_args
from aws_sagemaker_remote.training.main import sagemaker_training_run, sagemaker_env_args
from aws_sagemaker_remote.processing.args import sagemaker_processing_args
from aws_sagemaker_remote.processing.main import sagemaker_processing_handle
from .config import IgniterConfig
from aws_sagemaker_remote.commands import Command
from aws_sagemaker_remote.training.main import TrainingCommand
from pytorch_igniter.trainer import train, RunSpec
from .args import train_kwargs, bool_argument
from pytorch_igniter.evaluator import evaluate


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
    eval_event= 'EPOCH_COMPLETED'
):
    parser.add_argument(
        '--eval-event', default=eval_event, help='Evaluation event'
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
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                        help='random seed (default: 1)')


def model_args(parser, device=None):
    parser.add_argument("--device", type=str, default=device,
                        help="device to use (default: {})".format(device))


class TrainCommand(TrainingCommand):
    def runner(self, args):
        model = self.igniter_config.make_model(args).to(args.device)
        trainer = self.igniter_config.make_trainer(args, model).to(args.device)
        to_save = {
            'user_trainer': trainer,
            'model': model
        }
        #to_save['model'] = model
        train(
            to_save=to_save,
            train_spec=trainer.spec,
            # todo: check kwargs
            **train_kwargs(args),
            parameters=vars(args),
            # ,
            inference_args=args,
            inference_spec=self.igniter_config.inference_spec,
            model=model
            # =args
        )

    def argparse_callback(self, parser):
        group = parser.add_argument_group(
            title='Model',
            description='Model arguments'
        )
        model_args(group)
        if self.igniter_config.model_args:
            self.igniter_config.model_args(group)
        group = parser.add_argument_group(
            title='Training',
            description='Training arguments'
        )
        train_args(group)
        if self.igniter_config.train_args:
            self.igniter_config.train_args(group)
        parser.add_argument(
            '--cmd', default=self.cmd, help=argparse.SUPPRESS
        )
        group = parser.add_argument_group(
            title='MLflow',
            description='MLflow arguments'
        )
        mlflow_args(group)

    def __init__(self, igniter_config: IgniterConfig, script, cmd='train', **kwargs):
        super(TrainCommand, self).__init__(
            help='Train a model',
            main=self.runner,
            script=script,
            inputs=igniter_config.train_inputs,
            argparse_callback=self.argparse_callback,
            **kwargs
        )
        self.igniter_config = igniter_config
        self.train_config = None
        self.script = script
        self.cmd = cmd


class EvalCommand(TrainingCommand):
    def runner(self, args):
        model = self.igniter_config.make_model(args).to(args.device)
        evaluator = self.igniter_config.make_evaluator(
            args, model)
        evaluate(
            eval_spec=evaluator.spec,
            to_load={
                'model': model
            },
            model_dir=args.model_dir,
            output_dir=args.output_dir
            # todo: check kwargs
            # **train_kwargs(args),
            # parameters=vars(args)
        )

    def argparse_callback(self, parser):
        group = parser.add_argument_group(
            title='Model',
            description='Model arguments'
        )
        model_args(group)
        if self.igniter_config.model_args:
            self.igniter_config.model_args(group)
        group = parser.add_argument_group(
            title='Evaluation',
            description='Evaluation arguments'
        )
        if self.igniter_config.eval_args:
            self.igniter_config.eval_args(group)
        parser.add_argument(
            '--cmd', default=self.cmd, help=argparse.SUPPRESS
        )
        group = parser.add_argument_group(
            title='MLflow',
            description='MLflow arguments'
        )
        mlflow_args(group)

    def __init__(self, igniter_config: IgniterConfig, script, cmd='eval', **kwargs):
        super(EvalCommand, self).__init__(
            help='Evaluate a model',
            script=script,
            main=self.runner,
            inputs=igniter_config.eval_inputs,
            argparse_callback=self.argparse_callback,
            **kwargs
        )
        self.igniter_config = igniter_config
        self.script = script
        self.cmd = cmd

    # def configure(self, parser: argparse.ArgumentParser):
    #    super(EvalCommand, self).configure(parser)
    # def run(self, args):
    #    pass


class TrainAndEvalCommand(TrainingCommand):
    def runner(self, args):
        model = self.igniter_config.make_model(args).to(args.device)
        trainer = self.igniter_config.make_trainer(
            args, model).to(args.device)
        evaluator = self.igniter_config.make_evaluator(
            args, model)
        to_save = {
            'user_trainer': trainer,
            'model': model
        }
        train(
            to_save=to_save,
            model=model,
            train_spec=trainer.spec,
            eval_spec=evaluator.spec,
            # todo: check kwargs
            **train_kwargs(args),
            parameters=vars(args),
            inference_args=args,
            inference_spec=self.igniter_config.inference_spec,
        )

    def argparse_callback(self, parser):
        group = parser.add_argument_group(
            title='Model',
            description='Model arguments'
        )
        model_args(group)
        if self.igniter_config.model_args:
            self.igniter_config.model_args(group)
        group = parser.add_argument_group(
            title='Training',
            description='Training arguments'
        )
        train_args(group)
        if self.igniter_config.train_args:
            self.igniter_config.train_args(group)
        group = parser.add_argument_group(
            title='Evaluation',
            description='Evaluation arguments'
        )
        if self.igniter_config.eval_args:
            self.igniter_config.eval_args(group)
        group = parser.add_argument_group(
            title='Training and Evaluation',
            description='Training and evaluation arguments'
        )
        train_and_eval_args(group)
        parser.add_argument(
            '--cmd', default=self.cmd, help=argparse.SUPPRESS
        )
        group = parser.add_argument_group(
            title='MLflow',
            description='MLflow arguments'
        )
        mlflow_args(group)

    def __init__(self, igniter_config: IgniterConfig, script, cmd='train-and-eval', **kwargs):
        super(TrainAndEvalCommand, self).__init__(
            help='Train and evaluate a model',
            script=script,
            main=self.runner,
            argparse_callback=self.argparse_callback,
            inputs={
                **igniter_config.train_inputs,
                **igniter_config.eval_inputs
            },
            **kwargs
        )
        self.cmd = cmd
        self.igniter_config = igniter_config
        self.script = script

    # def configure(self, parser: argparse.ArgumentParser):
    #     super(TrainAndEvalCommand, self).configure(parser)
