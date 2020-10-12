
from .spec import RunSpec
from .engine import build_engine, Engine
from .metrics import SafeAverage
from .mlflow_ctx import mlflow_ctx, get_mlflow_logger
from .util import load_from, get_metrics
from tqdm import tqdm
import torch
import mlflow

def dummy_step(engine, batch):
    pass

def evaluate(
    eval_spec: RunSpec,
    output_dir=None,
    model_dir=None,
    to_load=None,
    tag='eval',
    mlflow_enable=True,
    mlflow_tracking_uri=None,
    trainer = None
):
    if mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    ctx = mlflow_ctx(output_dir=output_dir, mlflow_enable=mlflow_enable, allow_new=True)
    with ctx:
        if to_load is not None:
            dummy_trainer = Engine(dummy_step)
            to_load = {
                'trainer' : dummy_trainer,
                **to_load
            }
            path, iteration = load_from(model_dir=model_dir, to_load=to_load)
            tqdm.write("Loaded epoch {} iteration {} from {}".format(
                dummy_trainer.state.epoch,
                dummy_trainer.state.iteration, 
                path))
            assert iteration == dummy_trainer.state.iteration
            if trainer is None:
                trainer = dummy_trainer
        mlflow_logger = get_mlflow_logger(
            output_dir=output_dir,
            mlflow_enable=mlflow_enable
        )
        evaluator = build_engine(
            spec=eval_spec,
            output_dir=output_dir,
            tag=tag,
            metric_cls=SafeAverage,
            is_training=False,
            mlflow_logger=mlflow_logger,
            trainer=trainer
        )
        print("Metrics: {}".format(eval_spec.metrics))
        evaluator.run(
            eval_spec.loader, max_epochs=eval_spec.max_epochs, epoch_length=eval_spec.epoch_length
        )
        return get_metrics(engine=evaluator)
