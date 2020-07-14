import argparse
import os
import random
import warnings
import numpy as np

import re
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.utils.data as data

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Average

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import backward

LOGS_FNAME = "logs.tsv"
PLOT_FNAME = "plot.svg"
EVAL_MESSAGE = "[{epoch}/{max_epochs}][{i}/{max_i}][Evaluation]"
TRAIN_MESSAGE = "[{epoch}/{max_epochs}][{i}/{max_i}]"
LOADED = "Loaded {}, epoch {}, iteration {}"
COMPLETE = "Training complete"
INTERRUPTED = "KeyboardInterrupt caught. Exiting gracefully."


def get_metrics(engine, metric_names='all'):
    if metric_names == 'all':
        metrics = engine.state.metrics.items()
    else:
        metrics = [(metric, engine.state.metrics[metric])
                   for metric in metric_names]
    for name, value in metrics:
        if torch.is_tensor(value):
            yield name, value.item()
        else:
            yield name, value


def print_logs(engine, trainer=None, fmt=TRAIN_MESSAGE, metric_names='all'):
    if trainer is None:
        trainer = engine
    message = fmt.format(
        epoch=trainer.state.epoch,
        max_epochs=trainer.state.max_epochs,
        i=engine.state.iteration -
        ((engine.state.epoch-1)*engine.state.epoch_length),
        max_i=engine.state.epoch_length
    )
    for name, value in get_metrics(engine, metric_names=metric_names):
        message += " | {name}: {value}".format(
            name=name, value=str(round(value, 3)))
    tqdm.write(message)


def save_logs(engine, fname, trainer=None, metric_names='all'):
    if trainer is None:
        trainer = engine
    columns = ['iteration', 'epoch']
    values = [str(trainer.state.iteration), str(trainer.state.epoch)]
    for key, value in get_metrics(engine, metric_names=metric_names):
        columns.append(key)
        values.append(str(round(value, 5)))
    with open(fname, "a") as f:
        if f.tell() == 0:
            print("\t".join(columns), file=f)
        print("\t".join(values), file=f)


def create_plots(engine, logs_fname, plots_fname, metric_names='all'):
    try:
        import matplotlib as mpl

        mpl.use("agg")

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

    except ImportError:
        warnings.warn(
            "Loss plots will not be generated -- pandas or matplotlib not found")

    else:
        if os.path.exists(logs_fname):
            df = pd.read_csv(logs_fname,
                             delimiter="\t", index_col="iteration")
            if metric_names != 'all':
                df = df.filter(
                    items=metric_names,
                    axis='columns'
                )
            _ = df.plot(subplots=True, figsize=(20, 20))
            _ = plt.xlabel("Iteration number")
            fig = plt.gcf()
            fig.savefig(plots_fname)
            plt.close(fig)


def handle_exception(engine, e, callback=None, **kwargs):
    if isinstance(e, KeyboardInterrupt):
        engine.terminate()
        tqdm.write(INTERRUPTED)
        if callback is not None:
            callback(engine, **kwargs)
    else:
        raise e


def get_last_checkpoint(checkpoint_handler: ModelCheckpoint):
    dirname = checkpoint_handler.save_handler.dirname
    fmt = "{}checkpoint_(\\d+){}".format(checkpoint_handler._fname_prefix,
                                         checkpoint_handler._ext)

    def parse(fn):
        m = re.match(fmt, fn)
        if m:
            return int(m.group(1))
        else:
            return None
    files = list((fn, parse(fn)) for fn in os.listdir(dirname))
    files = filter(lambda x: x[1] is not None, files)
    files = sorted(files, key=lambda x: x[1])
    if files:
        last_file, last_iteration = files[-1]
        return os.path.join(dirname, last_file), last_iteration
    else:
        return None, None


def get_value(key):
    def fn(x):
        return x[key]
    return fn


def get_mean_value(key):
    def fn(x):
        return torch.mean(x[key])
    return fn


def auto_metric(value, cls=RunningAverage):
    if isinstance(value, str):
        return cls(output_transform=get_value(value))
    else:
        return value


def image_saver(engine, output_path, fn):
    path = output_path.format(engine.state.epoch)
    vutils.save_image(fn(engine).detach(), path)

def image_saver_callback(output_path, images):
    """
    Add a callback to 

    spec(
        callback=image_saver_callback('images',{'generated':get_value('generated')}),
        ...
    )
    """
    def fn(engine):
        for key, value in images.items():
            engine.add_event_handler(
                event_name=Events.EPOCH_COMPLETED,
                handler=image_saver,
                output_path=os.path.join(output_path, key),
                fn=value
            )
    return fn


def apply_to_tensors(tensors, fn):
    if torch.is_tensor(tensors):
        return fn(tensors)
    elif isinstance(tensors, list):
        return list(apply_to_tensors(tensor, fn) for tensor in tensors)
    elif isinstance(tensors, tuple):
        return tuple(apply_to_tensors(tensor, fn) for tensor in tensors)
    elif isinstance(tensors, dict):
        return dict((k, apply_to_tensors(tensor, fn)) for k, tensor in tensors.items())
    else:
        return tensors

def tensors_to_device(device):
    def fn(tensors):
        return apply_to_tensors(tensors=tensors, fn=lambda tensor: tensor.to(device))
    return fn

class RunSpec(object):
    @classmethod
    def train_spec(
        cls,
        loader,
        model,
        criteria,
        optimizer,
        preproc=None,
        **kwargs
    ):
        def step(engine, batch):
            model.train()
            model.zero_grad()
            if preproc is not None:
                batch = preproc(batch)
            inputs, targets = batch
            outputs = model(inputs)
            loss = criteria(outputs, targets)
            loss.backward()
            optimizer.step()
            return {
                "loss": loss
            }
        metrics = {'loss': 'loss'}
        return cls(
            loader=loader,
            step=step,
            metrics=metrics,
            **kwargs
        )

    @classmethod
    def eval_spec(
        cls,
        loader,
        model,
        criteria,
        preproc=None,
        **kwargs
    ):
        def step(engine, batch):
            model.eval()
            if preproc is not None:
                batch = preproc(batch)
            inputs, targets = batch
            outputs = model(inputs)
            loss = criteria(outputs, targets)
            return {
                "loss": loss
            }
        metrics = {'loss': 'loss'}
        return cls(
            loader=loader,
            step=step,
            metrics=metrics,
            **kwargs
        )

    def __init__(
        self,
        loader,
        step,
        metrics,
        max_epochs=1,
        epoch_length=None,
        callback=None,
        enable_pbar=True,
        pbar_metrics='all',
        print_event='default',
        print_metrics='all',
        print_fmt='default',
        log_event='default',
        log_metrics='all',
        plot_event='default',
        plot_metrics='all',
        enable_timer=True
    ):
        self.loader = loader
        self.step = step
        self.metrics = metrics
        self.max_epochs = max_epochs
        self.epoch_length = epoch_length
        self.callback = callback
        self.enable_pbar = enable_pbar
        self.pbar_metrics = pbar_metrics
        self.print_event = print_event
        self.print_metrics = print_metrics
        self.print_fmt = print_fmt
        self.log_event = log_event
        self.log_metrics = log_metrics
        self.plot_event = plot_event
        self.plot_metrics = plot_metrics
        self.enable_timer = enable_timer

    def set_defaults(self, is_training=True):
        """
        Fill in the default events for training or evaluation specs
        """
        if self.metrics is None:
            self.metrics = {}
        if self.plot_event == 'default':
            self.plot_event = Events.EPOCH_COMPLETED
        if is_training:
            # Log and print every 100 training iterations
            if self.log_event == 'default':
                self.log_event = Events.ITERATION_COMPLETED(every=100)
            if self.print_event == 'default':
                self.print_event = Events.ITERATION_COMPLETED(every=100)
            if self.print_fmt == 'default':
                self.print_fmt = TRAIN_MESSAGE
        else:
            # Log and print at the end of each evaluation
            if self.log_event == 'default':
                self.log_event = Events.EPOCH_COMPLETED
            if self.print_event == 'default':
                self.print_event = Events.EPOCH_COMPLETED
            if self.print_fmt == 'default':
                self.print_fmt = EVAL_MESSAGE


def timer_metric(engine, name='timer'):
    timer = Timer(average=True)
    timer.attach(
        engine,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED)

    def handler(_engine):
        _engine.state.metrics[name] = timer.value()
    engine.add_event_handler(
        event_name=Events.ITERATION_COMPLETED,
        handler=handler
    )


def build_engine(
    spec: RunSpec,
    output_dir,
    trainer=None,
    metric_cls=RunningAverage,
    prefix=""
):
    plot_fname = os.path.join(output_dir, "{}{}".format(prefix, PLOT_FNAME))
    logs_fname = os.path.join(output_dir, "{}{}".format(prefix, LOGS_FNAME))
    # Create engine
    engine = Engine(spec.step)
    if trainer is None:
        # training
        trainer = engine
        spec.set_defaults(is_training=True)
    else:
        # evaluation
        spec.set_defaults(is_training=False)

    # Attach metrics
    for name, metric in spec.metrics.items():
        auto_metric(metric, cls=metric_cls).attach(engine, name)
    if spec.enable_timer:
        timer_metric(engine=engine)

    # Progress bar
    if spec.enable_pbar:
        ProgressBar().attach(engine, metric_names=spec.pbar_metrics)

    # Print logs
    if spec.print_event is not None:
        engine.add_event_handler(
            event_name=spec.print_event,
            handler=print_logs,
            trainer=trainer,
            fmt=spec.print_fmt,
            metric_names=spec.print_metrics)

    # Save logs
    if spec.log_event is not None:
        engine.add_event_handler(
            event_name=spec.log_event,
            handler=save_logs,
            fname=logs_fname,
            trainer=trainer,
            metric_names=spec.log_metrics)

    # Plot metrics
    if spec.plot_event is not None:
        # Plots require logs
        assert spec.log_event is not None
        engine.add_event_handler(
            event_name=spec.plot_event,
            handler=create_plots,
            logs_fname=logs_fname,
            plots_fname=plot_fname,
            metric_names=spec.plot_metrics)

    # Optional user callback for additional configuration
    if spec.callback is not None:
        spec.callback(engine)

    return engine


def train(
    to_save,
    output_dir,
    train_spec: RunSpec,
    eval_spec: RunSpec,
    eval_event=Events.EPOCH_COMPLETED,
    save_event=Events.EPOCH_COMPLETED,
    n_saved=10
):
    # Create trainer

    trainer = build_engine(
        spec=train_spec,
        output_dir=output_dir
    )
    to_save = {'trainer': trainer, **to_save}

    # Saver
    checkpoint_handler = ModelCheckpoint(
        output_dir, filename_prefix="", n_saved=n_saved, require_empty=False)
    trainer.add_event_handler(
        event_name=save_event,
        handler=checkpoint_handler,
        to_save=to_save
    )

    # Optional evaluation
    if eval_spec is not None:
        assert eval_event is not None
        if not isinstance(eval_spec, dict):
            eval_spec = {
                'eval': eval_spec
            }
        # Build evaluators
        evaluators = [
            (
                build_engine(
                    spec=spec,
                    output_dir=output_dir,
                    prefix=prefix+"-",
                    trainer=trainer,
                    metric_cls=Average
                ),
                spec
            )
            for prefix, spec in eval_spec.items()
        ]
        # Add evaluation hook to trainer

        def evaluation(engine):
            for evaluator, spec in evaluators:
                evaluator.run(
                    spec.loader,
                    max_epochs=spec.max_epochs,
                    epoch_length=spec.epoch_length)
        trainer.add_event_handler(
            event_name=eval_event,
            handler=evaluation)

    # Handle ctrl-C or other exceptions
    def exception_callback(engine):
        # Save on exit
        if engine.state.iteration and engine.state.iteration > 0:
            _, last_iteration = get_last_checkpoint(
                checkpoint_handler=checkpoint_handler)
            if last_iteration is None or last_iteration < engine.state.iteration:
                checkpoint_handler(engine=engine, to_save=to_save)
    trainer.add_event_handler(
        event_name=Events.EXCEPTION_RAISED,
        handler=handle_exception,
        callback=exception_callback
    )

    # Get last checkpoint
    checkpoint, _ = get_last_checkpoint(checkpoint_handler)
    if checkpoint:
        # Load checkpoint
        loaded = torch.load(checkpoint)
        for key, value in to_save.items():
            value.load_state_dict(loaded[key])
        tqdm.write(LOADED.format(
            checkpoint, trainer.state.epoch, trainer.state.iteration))
        if Engine._is_done(trainer.state):
            # Training complete
            tqdm.write(COMPLETE)
        else:
            # Continue training
            trainer.run(train_spec.loader)
    else:
        # Start training
        trainer.run(
            train_spec.loader,
            max_epochs=train_spec.max_epochs,
            epoch_length=train_spec.epoch_length)
