import json

def get_hyperparameters(hyperparameters_file='hyperparameters.json'):
    with open(hyperparameters_file, 'r') as f:
        hyperparameters = json.load(f)
    return hyperparameters
