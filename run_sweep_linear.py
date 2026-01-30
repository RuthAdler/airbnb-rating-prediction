import itertools
import subprocess
import sys

models_no_depth = ["dummy", "linear_regression", "ridge", "lasso"]
scalers = ["standard", "robust", "minmax"]
combinations = list(itertools.product(models_no_depth, scalers))

# Execution loop
for model, scaler in combinations:
    cmd = [
        sys.executable, "run_experiment.py",
            "--team_member", "Ella Yakir",
            "--model", model,
            "--scaler", scaler,
            "--dataset_version", "v1",
            "--notes", "Sweep_Run"
        ]

        
    print(f"Running experiment with model: {model}, scaler: {scaler}")
    subprocess.run(cmd)