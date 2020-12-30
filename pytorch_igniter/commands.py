import argparse

from aws_sagemaker_remote.training.args import sagemaker_training_args
from aws_sagemaker_remote.training.main import sagemaker_training_run, sagemaker_env_args
from aws_sagemaker_remote.processing.args import sagemaker_processing_args
from aws_sagemaker_remote.processing.main import sagemaker_processing_handle
from aws_sagemaker_remote.commands import Command
from aws_sagemaker_remote.training.main import TrainingCommand

from .config import IgniterConfig
from pytorch_igniter.trainer import train, RunSpec
from .args import train_kwargs, bool_argument
from pytorch_igniter.evaluator import evaluate
from pytorch_igniter.args import fix_device, add_eval_args, add_mlflow_args, add_model_args, add_train_args, add_tran_and_eval_args
import torch


class TrainCommand(TrainingCommand):
    def runner(self, args):
        args = fix_device(args)
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
        add_model_args(parser, self.igniter_config.model_args)
        add_train_args(parser, self.igniter_config.train_args)
        add_mlflow_args(parser)
        parser.add_argument(
            '--cmd', default=self.cmd, help=argparse.SUPPRESS
        )

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
        args = fix_device(args)
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
        add_model_args(parser, self.igniter_config.model_args)
        add_eval_args(parser, self.igniter_config.train_args)
        # add_mlflow_args(parser)
        parser.add_argument(
            '--cmd', default=self.cmd, help=argparse.SUPPRESS
        )

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
        args = fix_device(args)
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
        add_model_args(parser, self.igniter_config.model_args)
        add_train_args(parser, self.igniter_config.train_args)
        add_eval_args(parser, self.igniter_config.eval_args)
        add_tran_and_eval_args(parser)
        add_mlflow_args(parser)
        parser.add_argument(
            '--cmd', default=self.cmd, help=argparse.SUPPRESS
        )

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
