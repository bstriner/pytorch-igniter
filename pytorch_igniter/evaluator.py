
from .spec import RunSpec
from .engine import build_engine, Engine
from .metrics import SafeAverage
from .util import load_from, get_metrics
from tqdm import tqdm
import torch

def dummy_step(engine, batch):
    pass

def evaluate(
    eval_spec: RunSpec,
    output_dir=None,
    to_load=None,
    tag='eval',
    mlflow_logger=None
):
    if to_load is not None:
        trainer = Engine(dummy_step)
        to_load = {
            'trainer' : trainer,
            **to_load
        }
        path, iteration = load_from(output_dir=output_dir, to_load=to_load)
        tqdm.write("Loaded epoch {} iteration {} from {}".format(
            trainer.state.epoch,
            trainer.state.iteration, 
            path))
    else:
        trainer = None

    evaluator = build_engine(
        spec=eval_spec,
        output_dir=output_dir,
        tag=tag,
        metric_cls=SafeAverage,
        is_training=False,
        mlflow_logger=mlflow_logger,
        trainer=trainer
    )
    evaluator.run(
        eval_spec.loader, max_epochs=eval_spec.max_epochs, epoch_length=eval_spec.epoch_length
    )
    return get_metrics(engine=evaluator)
