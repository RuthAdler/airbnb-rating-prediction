import itertools
import subprocess
import sys

models_no_depth = ["dummy", "linear_regression", "ridge", "lasso"]
models_with_depth = ["decision_tree", "random_forest", "xgboost"]
scalers = ["standard", "robust", "minmax"]
depths = [None, 10, 20] 
combinations = list(itertools.product(models_with_depth, scalers, depths)) + list(itertools.product(models_no_depth, scalers, [None]))

# Execution loop
for model, scaler, depth in combinations:
    cmd = [
        sys.executable, "run_experiment.py",
            "--team_member", "Ella Yakir",
            "--model", model,
            "--scaler", scaler,
            "--dataset_version", "v1",
            "--notes", "Sweep_Run"
        ]
        
    if depth:
        cmd.extend(["--max_depth", str(depth)])
    print(f"Running experiment with model: {model}, scaler: {scaler}, max_depth: {depth}")
    subprocess.run(cmd)