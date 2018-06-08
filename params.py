import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, help='an experiment name to override settings', required=False)
parser.add_argument('--restore', type=str, help='previous run to restore from', required=False)
args = parser.parse_args()

def load_params():
    if args.restore:
       with open("results/"+args.restore+"/params.json") as f:
           params = json.load(f)
           params["run_id"] = args.restore
           params["restore"] = True
           return params

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