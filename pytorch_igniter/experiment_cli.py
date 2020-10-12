from pytorch_igniter.config import IgniterConfig
import argparse
from aws_sagemaker_remote.training.args import sagemaker_training_args
from aws_sagemaker_remote.training.main import sagemaker_training_run, sagemaker_env_args
from pytorch_igniter.commands import TrainCommand, TrainAndEvalCommand, EvalCommand
from aws_sagemaker_remote.commands import run_commands
import sys
import csv
import os
from aws_sagemaker_remote.training.args import sagemaker_env_arg

IGNITER_COMMAND='IGNITER_COMMAND'

def experiment_cli_commands(
    script,
    config: IgniterConfig,
    extra_commands=None,
    **kwargs
):
    commands = {
        'train':TrainCommand(igniter_config=config, script=script, **kwargs),
        'eval':EvalCommand(igniter_config=config, script=script, **kwargs),
        'train-and-eval':TrainAndEvalCommand(igniter_config=config, script=script, **kwargs)
    }
    if extra_commands:
        commands.update(extra_commands)
    return commands


"""
def experiment_sagemaker(args, train_config, main):
    if args.sagemaker_run:
        sagemaker_training_run(
            args=args,
            config=train_config
        )
    else:
        args = sagemaker_env_args(args=args, config=train_config)
        main(args)
"""


def experiment_cli(
    script,
    config: IgniterConfig,
    extra_commands=None,
    description=None,
    dry_run=False,
    **kwargs
):
    # dest='command' todo: variable destination
    commands = experiment_cli_commands(
        script=script,
        config=config,
        extra_commands=extra_commands,
        **kwargs)

    argv = []
    cmd = os.getenv(IGNITER_COMMAND, None)
    if cmd:
        cmd = next(csv.reader([cmd]))
        if cmd:
            argv.extend(cmd)
    cmd = sagemaker_env_arg()
    if cmd:
        cmd = cmd.get('hyperparameters', {}).get('cmd',None)
        if cmd:
            cmd = next(csv.reader([cmd]))
            if cmd:
                argv.extend(cmd)
    argv.extend(sys.argv[1:])
    return run_commands(
        commands=commands,
        description=description,
        argv=argv,
        dry_run=dry_run
    )
