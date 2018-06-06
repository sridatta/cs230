import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, help='an experiment name to override settings', required=False)
parser.add_argument('--sweep', type=int, help='number of random hyperparameter sweeps to make', required=False)
args = parser.parse_args()

def load_params():
    with open("params.json") as f:
        params = json.load(f)

    with open("data/dataset_params.json") as f:
        params.update(json.load(f))

    params["experiment"] = "base"
    if args.experiment:
        params["experiment"] = args.experiment
        with open("experiments/"+args.experiment+".json") as f:
            exp = json.load(f)
            params.update(exp)
    return params