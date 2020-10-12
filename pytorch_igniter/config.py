

class IgniterConfig(object):
    def __init__(
            self,
            model_args=None,
            train_args=None,
            eval_args=None,
            description=None,
            train_inputs=None,
            eval_inputs=None,
            make_model=None,
            make_evaluator=None,
            make_trainer=None,
            inference_spec=None,
            **igniter_args
    ):
        self.model_args = model_args
        self.train_args = train_args
        self.eval_args = eval_args
        self.description = description or "Experiment CLI"
        self.train_inputs = train_inputs or {}
        self.eval_inputs = eval_inputs or {}
        self.make_model = make_model
        self.make_evaluator = make_evaluator
        self.make_trainer = make_trainer
        self.inference_spec = inference_spec
        self.igniter_args = igniter_args
