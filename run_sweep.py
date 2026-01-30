import itertools

models_no_depth = ["dummy", "linear_regression", "ridge", "lasso"]
models_with_depth = ["decision_tree", "random_forest", "xgboost"]
scalers = ["standard", "robust", "minmax"]
depths = [None, 10, 20] 
team_member = "Ella Yakir" 
dataset_version = "v1"
combinations = list(itertools.product(models_with_depth, scalers, depths)) + list(itertools.product(models_no_depth, scalers, [None]))

