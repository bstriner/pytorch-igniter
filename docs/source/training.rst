Training
========

``pytorch-igniter`` constructs a training engine so you can focus on machine learning.

Features
+++++++++

* Only create a model and write functions that train and evaluate on a single batch. ``pytorch-igniter`` constructs
  the training engine that checkpoints your model while training, evaluating, and logging.
* Standardized and documented ``argparse`` command-line arguments like ``--batch-size``, ``--max-epochs``, and ``--learning-rate``.
  Only write custom arguments that are unique to your script.
* Save model on ``ctrl-C`` or ``kill``. Automatically resume model from latest checkpoint. Configurable checkpointing.
* Simplify defining metrics. Metrics can average or otherwise accumulate data and can be saved, printed, and more
  depending on configuration.
* Integrate with `MLflow <https://mlflow.org/>`_ for tracking training runs, including hyperparameters and metrics.
* Integrate with AWS SageMaker using `aws-sagemaker-remote <https://aws-sagemaker-remote.readthedocs.io/en/latest/>`_
  for tracking training runs and executing training remotely on managed containers.


Basic Usage
+++++++++++

.. code-block:: python
  
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms

    from pytorch_igniter import train, RunSpec
    from pytorch_igniter.args import train_kwargs
    from pytorch_igniter.main import igniter_main


    def main(
        args
    ):

        # Create data loaders
        train_loader = DataLoader(...)
        eval_loader = DataLoader(...)

        # Create model, optimizer, and criteria
        model = nn.Sequential(...)
        optimizer = torch.optim.Adam(...)
        criteria = nn.CrossEntropyLoss(...)

        # Single step of training
        def train_step(engine, batch):
            # Do training
            model.train()
            model.zero_grad()
            inputs, labels = batch
            outputs = model(inputs)
            loss = criteria(input=outputs, target=labels)
            loss.backward()
            optimizer.step()
            return {
                "loss": loss
            }

        # Single step of evaluation
        def eval_step(engine, batch):
            # Do evaluation
            model.eval()
            inputs, labels = batch
            outputs = model(inputs)
            loss = criteria(input=outputs, target=labels)
            return {
                "loss": loss
            }

        # Metrics average the outputs of the step functions and are printed and saved to logs
        metrics = {
            'loss': 'loss'
        }

        # Objects to save
        to_save = {
            "model": model,
            "optimizer": optimizer
        }

        train(
            to_save=to_save,
            # Training setup
            train_spec=RunSpec(
                step=train_step,
                loader=train_loader,
                metrics=metrics
            ),
            # Evaluation setup
            eval_spec=RunSpec(
                step=eval_step,
                loader=eval_loader,
                metrics=metrics
            ),
            **train_kwargs(args),
            parameters=vars(args)
        )


    if __name__ == "__main__":
        igniter_main(
            main=main,
            inputs={
                'data': 'data'
            },
            # ...
        )

Command-Line Arguments
++++++++++++++++++++++

Note that additional command-line arguments are generated for each item in ``inputs`` and ``dependencies`` function arguments.

.. argparse::
   :module: pytorch_igniter.args
   :func: parser_for_docs
   :prog: pytorch-igniter

See ``aws-sagemaker-remote`` documentation for SageMaker option documentation.