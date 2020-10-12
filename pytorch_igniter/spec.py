
from ignite.engine import Events
from .util import TRAIN_MESSAGE, EVAL_MESSAGE

class InferenceSpec(object):
    def __init__(self,
    inferencer,
    requirements=None,
    dependencies=None,
    input_fn=None,
    output_fn=None):
        self.inferencer=inferencer
        self.requirements=requirements
        self.dependencies=dependencies or []
        self.input_fn = input_fn or "from pytorch_igniter.inference.image_input_fn import input_fn"
        self.output_fn = output_fn or "from pytorch_igniter.inference.json_output_fn import output_fn"


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
        print_metric_fmt=' | {name}: {value}',
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
        self.print_metric_fmt = print_metric_fmt
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
