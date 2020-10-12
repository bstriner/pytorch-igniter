
from collections import OrderedDict
import torch
from ignite.engine import Events
import os
import torchvision.utils as vutils
import re
from tqdm import tqdm
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from torch import nn
import signal
from contextlib import contextmanager
import numpy as np
import warnings

EVAL_MESSAGE = "[{epoch}/{max_epochs}][{i}/{max_i}][Evaluation]"
TRAIN_MESSAGE = "[{epoch}/{max_epochs}][{i}/{max_i}]"
INTERRUPTED = "KeyboardInterrupt caught. Exiting gracefully."


class StateDictStorage(nn.Module):
    def __init__(self):
        super(StateDictStorage, self).__init__()
        self._state_dict = None

    def load_state_dict(self, state_dict, strict=True):
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict


def get_metrics(engine, metric_names='all'):
    if metric_names == 'all':
        metrics = engine.state.metrics
    else:
        metrics = OrderedDict([
            (metric_name, engine.state.metrics[metric_name])
            for metric_name in metric_names
        ])
    metrics = tensors_to_numpy(metrics)
    metrics = OrderedDict([
        (k, np.array(v).item())
        for k, v
        in metrics.items()
        if np.array(v).size == 1
    ])
    return metrics


def print_logs(engine, trainer=None, fmt=TRAIN_MESSAGE, metric_fmt=" | {name}: {value}",
               metric_names='all'):
    if trainer is None:
        trainer = engine
    message = fmt.format(
        epoch=trainer.state.epoch,
        max_epochs=trainer.state.max_epochs,
        i=engine.state.iteration -
        ((engine.state.epoch-1)*engine.state.epoch_length),
        max_i=engine.state.epoch_length
    )
    for name, value in get_metrics(engine, metric_names=metric_names).items():
        message += metric_fmt.format(
            name=name, value=str(round(value, 3)))
    tqdm.write(message)


def save_logs(engine, fname, trainer=None, metric_names='all'):
    if trainer is None:
        trainer = engine
    columns = ['iteration', 'epoch']
    values = [str(trainer.state.iteration), str(trainer.state.epoch)]
    for key, value in get_metrics(engine, metric_names=metric_names).items():
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


# todo: add sigint handling
def handle_exception(engine, e, callback=None, **kwargs):
    if isinstance(e, KeyboardInterrupt):
        engine.terminate()
        tqdm.write(INTERRUPTED)
        if callback is not None:
            callback(engine, **kwargs)
    else:
        raise e


def chain_callbacks(callbacks=None, **kwargs):
    if callbacks is not None:
        if isinstance(callbacks, (list, tuple)):
            for callback in callbacks:
                if callback is not None:
                    callback(**kwargs)
        else:
            callback(**kwargs)


def kill_signals():
    signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, 'SIGHUP'):
        signals.append(signal.SIGHUP)
    return signals


@contextmanager
def capture_signals(signals=None, callback=None,die=False, **kwargs):
    if signals is None:
        signals = kill_signals()
    original_handlers = [signal.getsignal(sig) for sig in signals]

    def handle_signal(sig, frame):
        if callback is not None:
            callback(**kwargs)
        raise KeyboardInterrupt("Received signal {}".format(sig))

    for sig in signals:
        signal.signal(sig, handle_signal)
    try:
        yield
    except KeyboardInterrupt as e:
        tqdm.write(e)
        if die:
            raise e
    finally:
        # note: only works if old handler was originally installed by Python
        for sig, original_handler in zip(signals, original_handlers):
            signal.signal(sig, original_handler)


def load_from(model_dir, to_load):
    path, iteration = find_last_checkpoint(model_dir)
    if path is None:
        raise FileNotFoundError("No checkpoints found in {}".format(path))
    loaded = torch.load(path)
    for k, v in to_load.items():
        v.load_state_dict(loaded[k])
    return path, iteration


def find_last_checkpoint(output_dir):
    checkpoint_handler = ModelCheckpoint(
        output_dir, filename_prefix="", require_empty=False)
    return get_last_checkpoint(checkpoint_handler=checkpoint_handler)


def get_last_checkpoint(checkpoint_handler: ModelCheckpoint):
    dirname = checkpoint_handler.save_handler.dirname
    if hasattr(checkpoint_handler, '_fname_prefix'):
        filename_prefix = checkpoint_handler._fname_prefix
    elif hasattr(checkpoint_handler, "filename_prefix"):
        filename_prefix = checkpoint_handler.filename_prefix
    else:
        filename_prefix = ""
    if hasattr(checkpoint_handler, '_ext'):
        ext = checkpoint_handler._ext
    elif hasattr(checkpoint_handler, "ext"):
        ext = checkpoint_handler.ext
    else:
        ext = "pt"
    if not ext.startswith("."):
        ext = "\\.{}".format(ext)
    fmt = "{}checkpoint_(\\d+){}".format(filename_prefix, ext)

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
    Add a callback to save images

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
        return OrderedDict((k, apply_to_tensors(tensor, fn)) for k, tensor in tensors.items())
    else:
        return tensors


def tensors_to_device(device):
    def fn(tensors):
        return apply_to_tensors(tensors=tensors, fn=lambda tensor: tensor.to(device))
    return fn


def tensors_to_items(tensors):
    return apply_to_tensors(tensors=tensors, fn=lambda tensor: tensor.item())


def tensors_to_numpy(tensors):
    return apply_to_tensors(tensors=tensors, fn=lambda tensor: tensor.detach().cpu().numpy())


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


if __name__=='__main__':
    with capture_signals():
        import time
        time.sleep(200)
    print("Done")