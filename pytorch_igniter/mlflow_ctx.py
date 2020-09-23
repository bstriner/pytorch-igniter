import os
import mlflow
import contextlib
from ignite.contrib.handlers.mlflow_logger import MLflowLogger
import yaml


class NullContext(object):
    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        pass


RUN_FNAME = 'run.yaml'


def mlflow_ctx(
    output_dir=None,
    run_id=None,
    mlflow_enable=True,
    allow_new=True,
    experiment_name=None,
    run_name=None,
    parameters=None
):
    if mlflow_enable:
        # Check for a run already in progress
        active_run = mlflow.active_run()
        if active_run is not None:
            print("MLflow ID {} active".format(active_run.info.run_id))
            return NullContext(), output_dir
        # Use argument
        if run_id is not None:
            print("MLflow ID {} set".format(run_id))
            return mlflow.start_run(run_id=run_id), output_dir
        # Check environment variable
        if 'MLFLOW_RUN_ID' in os.environ:
            run_id = os.environ['MLFLOW_RUN_ID']
            output_dir = os.path.join(output_dir, run_id)
            print("MLflow ID {} from environment".format(run_id))
            return mlflow.start_run(run_id=run_id), output_dir
        # Check saved run id
        if output_dir is not None:
            run_fname = os.path.join(output_dir, RUN_FNAME)
            if os.path.exists(run_fname):
                with open(run_fname) as f:
                    run_id = yaml.load(f, Loader=yaml.SafeLoader)[
                        'info']['run_id']
                print("MLflow ID {} from run file".format(run_id))
                return mlflow.start_run(run_id=run_id), output_dir
        # New run
        if allow_new:
            print("MLflow new run")
            if experiment_name:
                experiment = mlflow.get_experiment_by_name(
                    name=experiment_name)
                if experiment:
                    experiment_id = experiment.experiment_id
                else:
                    experiment_id = mlflow.create_experiment(
                        name=experiment_name)
            else:
                experiment_id = None
            ctx = mlflow.start_run(
                run_id=run_id, experiment_id=experiment_id, run_name=run_name)
            if parameters:
                mlflow.log_params(parameters)
            return ctx, output_dir
        else:
            raise ValueError("No existing MLflow run found")
    else:
        return NullContext(), output_dir


def get_mlflow_logger(output_dir, mlflow_enable):
    if mlflow_enable:
        mlflow_logger = MLflowLogger()
        if output_dir is not None:
            run_fname = os.path.join(output_dir, RUN_FNAME)
            if not os.path.exists(run_fname):
                active_run = mlflow.active_run()
                active_run = mlflow.get_run(active_run.info.run_id)
                with open(run_fname, 'w') as f:
                    yaml.dump(active_run.to_dictionary(), f)
        return mlflow_logger
    else:
        return None
