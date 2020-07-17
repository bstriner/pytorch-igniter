import mlflow

import os
import sys

run = mlflow.active_run()
print(run)
run_id = os.environ['MLFLOW_RUN_ID']
print(run_id)
print(sys.argv)
run = mlflow.get_run(run_id=run_id)
print(run)
print(run.info)
print(run.data.params)

print("hello")