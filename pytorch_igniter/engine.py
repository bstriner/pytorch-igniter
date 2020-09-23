
from .spec import RunSpec
import os
from ignite.metrics import RunningAverage
from ignite.engine import Engine, Events
from .util import chain_callbacks, auto_metric, timer_metric, print_logs, save_logs, create_plots, tensors_to_device
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.mlflow_logger import OutputHandler, global_step_from_engine
LOGS_FNAME = "logs.tsv"
PLOT_FNAME = "plot.svg"


def build_engine(
    spec: RunSpec,
    output_dir=None,
    trainer=None,
    metric_cls=RunningAverage,
    tag="",
    mlflow_logger=None,
    is_training=None,
    device=None
):
    if spec.plot_event is not None or spec.log_event is not None:
        assert output_dir is not None
        plot_fname = os.path.join(output_dir, "{}-{}".format(tag, PLOT_FNAME))
        logs_fname = os.path.join(output_dir, "{}-{}".format(tag, LOGS_FNAME))
    else:
        plot_fname = None
        logs_fname = None
    # Create engine
    if device:
        to_device = tensors_to_device(device=device)

        def step(engine, batch):
            batch = to_device(batch)
            return spec.step(engine, batch)
    else:
        step = spec.step
    engine = Engine(step)
    if trainer is None:
        # training
        trainer = engine
        if is_training is None:
            is_training = True
    else:
        # evaluation
        if is_training is None:
            is_training = False
    spec.set_defaults(is_training=is_training)

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
            metric_fmt=spec.print_metric_fmt,
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
    chain_callbacks(
        callbacks=spec.callback,
        engine=engine,
        trainer=trainer)

    if mlflow_logger is not None and spec.log_event is not None:
        mlflow_logger.attach(
            engine,
            log_handler=OutputHandler(
                tag=tag,
                metric_names=spec.log_metrics,
                global_step_transform=global_step_from_engine(trainer)
            ),
            event_name=spec.log_event
        )

    return engine
