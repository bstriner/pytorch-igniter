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
    checkpoint_dir=None,
    output_dir=None,
    run_id=None,
    mlflow_enable=True,
    allow_new=True,
    experiment_name=None,
    run_name=None,
    parameters=None,
    is_sagemaker=None,
    sagemaker_job_name=None
):
    if mlflow_enable:
        # Check for a run already in progress
        active_run = mlflow.active_run()
        if active_run is not None:
            print("MLflow ID {} active".format(active_run.info.run_id))
            return NullContext()
        # Use argument
        if run_id is not None:
            print("MLflow ID {} set".format(run_id))
            return mlflow.start_run(run_id=run_id)
        # Check saved run id
        if checkpoint_dir is not None:
            run_fname = os.path.join(checkpoint_dir, RUN_FNAME)
            if os.path.exists(run_fname):
                with open(run_fname) as f:
                    run_id = yaml.load(f, Loader=yaml.SafeLoader)[
                        'info']['run_id']
                print("MLflow ID {} from run file".format(run_id))
                return mlflow.start_run(run_id=run_id)
        # New run
        if allow_new:
            print("MLflow new run")
            if experiment_name:
                try:
                    experiment = mlflow.get_experiment_by_name(
                        name=experiment_name)
                    if experiment:
                        experiment_id = experiment.experiment_id
                    else:
                        experiment_id = mlflow.create_experiment(
                            name=experiment_name)
                except Exception as e:
                    print(e)
                    experiment_id = mlflow.create_experiment(
                            name=experiment_name)
                    
                #todo: wait for experiment to be fully created. otherwise start_run fails
            else:
                experiment_id = None
            ctx = mlflow.start_run(
                run_id=run_id, experiment_id=experiment_id, run_name=run_name)
            if parameters:
                mlflow.log_params(parameters)
            if is_sagemaker and sagemaker_job_name:
                mlflow.set_tag('SageMakerJobName', sagemaker_job_name)
            return ctx
        else:
            raise ValueError("No existing MLflow run found")
    else:
        return NullContext()


def get_mlflow_logger(output_dir=None, checkpoint_dir=None, mlflow_enable=True):
    if mlflow_enable:
        mlflow_logger = MLflowLogger()
        active_run = mlflow.active_run()
        active_run = mlflow.get_run(active_run.info.run_id)
        if output_dir is not None:
            run_fname = os.path.join(output_dir, RUN_FNAME)
            with open(run_fname, 'w') as f:
                yaml.dump(active_run.to_dictionary(), f)
        if checkpoint_dir is not None and output_dir != checkpoint_dir:
            run_fname = os.path.join(checkpoint_dir, RUN_FNAME)
            with open(run_fname, 'w') as f:
                yaml.dump(active_run.to_dictionary(), f)
        return mlflow_logger
    else:
        return None
