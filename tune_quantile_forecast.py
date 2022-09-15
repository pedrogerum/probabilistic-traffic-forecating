from ast import arguments
import os

import ray
from gluonts.env import env
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
import json
from quantile_forecast import *

env._push(use_tqdm=False)

def make_objective(paths, prediction_length, monot):
    def objective(config):
        losses = 0
        avg_crps = 0
        n = 0
        eighty_cover = 0
        ninetyfive_cover = 0
        for path in paths:
            print(path)
            dataset = load_dataset(path)
            quantiles = make_quantiles(dataset, config["num_quantiles"], 1e-3)
            trainer = make_trainer(config)
            estimator = make_mqrnn_estimator(trainer, quantiles, monot, dataset.metadata.freq, prediction_length = prediction_length)
            predictor = make_predictor(estimator, dataset)
            metrics, _ , _= evaluate(predictor, dataset, quantiles, prediction_length = prediction_length)
            losses += np.abs(metrics["60cover"] - 0.60)
            losses += np.abs(metrics["80cover"] - 0.80)
            losses += np.abs(metrics["95cover"] - 0.95)
            avg_crps += metrics["crps"]
            eighty_cover += metrics["80cover"]
            ninetyfive_cover += metrics["95cover"]
            n += 1
        tune.report(
            coverage_error = losses, 
            avg_crps = avg_crps / n,
            eighty_cover = eighty_cover / n,
            ninetyfive_cover = ninetyfive_cover / n
        )
    return objective

def absoluteFilePaths(directory):
    for f in os.listdir(directory):
        yield os.path.abspath(os.path.join(directory, f))

def main(args):
    import os
    ray.shutdown()
    ray.init(num_cpus=4, object_store_memory=78643200)
    paths = []
    for d in args.cv_dir:
        paths += list(absoluteFilePaths(d))
    print(paths)
    prediction_length = args.pred_length
    objective = make_objective(paths, prediction_length, args.monot)
    search_alg = OptunaSearch(
        mode="min", 
        metric="avg_crps"
    )

    analysis = tune.run(
    objective,
    config={
        "epochs": tune.randint(200, min(500, int(8000/prediction_length))),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "learning_rate_decay_factor": tune.uniform(0.25, 0.75),
        "weight_decay": tune.loguniform(1e-9, 1e-1),
        "num_quantiles": tune.randint(15,100),
    },
    metric="avg_crps",
    mode="min",
    search_alg=search_alg,
    num_samples=50)

    best_trial = analysis.get_best_trial("avg_crps", 'min', "all")
    best_config = best_trial.config

    print(best_trial)
    print(best_config)

    with open(args.save_path, "w+") as f:
        json.dump(best_config, f)

    try:
        results_dict = {}
        results_dict["best_CRPS"] = best_trial.avg_CRPS
        results_dict["best_80coverage"] = best_trial.eighty_cover
        results_dict["best_95coverage"] = best_trial.ninetyfive_cover
        print(results_dict["best_CRPS"])
        print(results_dict["best_80coverage"])
        print(results_dict["best_95coverage"])

        with open(args.results_path, "w+") as f:
            json.dump(results_dict, f)

    except:
        print("HAVING A PROBLEM HERE PRINTING BEST RESULTS")
        pass



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('cv_dir', type=str, nargs='+',
                        help='Path to dataset')
    parser.add_argument('--save_path', type=str,
                        help='Path to save config')
    parser.add_argument('--results_path', type=str,
                        help='Path to save results')
    parser.add_argument('--pred_length', type=int,
                        help='Prediction Length')
    parser.add_argument('--monot', type=bool,
                        help='is monotonicity enforced')
    args = parser.parse_args()
    main(args)
