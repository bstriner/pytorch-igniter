
from .spec import RunSpec
from .engine import build_engine
from ignite.metrics import Average


def evaluate(
    eval_spec: RunSpec,
    output_dir=None,
    tag='eval'
):
    evaluator = build_engine(
        spec=eval_spec,
        output_dir=output_dir,
        tag=tag,
        metric_cls=Average,
        is_training=False
    )
    evaluator.run(
        eval_spec.loader, max_epochs=eval_spec.max_epochs, epoch_length=eval_spec.epoch_length
    )
    return evaluator.state.metrics
