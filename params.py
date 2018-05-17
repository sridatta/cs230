import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, help='an experiment name to override settings', required=False)
args = parser.parse_args()

def load_params():
    with open("params.json") as f:
        params = json.load(f)

    with open("data/dataset_params.json") as f:
        params.update(json.load(f))

    if args.experiment:
        with open("experiment/"+args.experiment+".json") as f:
            exp = json.load(f)
            params.update(exp)
    return params