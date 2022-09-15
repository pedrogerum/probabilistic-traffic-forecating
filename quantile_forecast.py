import numpy as np
import os
import mxnet as mx
import pandas as pd
from mxnet import gluon
import gc 

from gluonts.env import env

env._push(use_tqdm=False)

import json
from gluonts.dataset.common import load_datasets
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.rolling_dataset import NumSplitsStrategy, StepStrategy, generate_rolling_dataset, truncate_features
from gluonts.model.seq2seq import MQRNNEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.distribution import PiecewiseLinearOutput
from gluonts.mx.trainer import Trainer
from gluonts.mx.trainer.model_averaging import SelectNBestMean
from gluonts.mx.trainer.model_iteration_averaging import Alpha_Suffix
from gluonts.evaluation import Evaluator
from pathlib import Path
from gluonts.evaluation.backtest import make_evaluation_predictions
from tqdm.notebook import tqdm
import warnings


from gluonts.env import env

env._push(use_tqdm=False)

from scipy import interpolate
def crps(tau, F_inv, x, n_steps=100000, miny=0, maxy=100):
    tau = np.concatenate((np.array([0]), tau, np.array([1])))
    F_inv = np.concatenate((np.array([miny]), F_inv, np.array([maxy])))
    F = interpolate.interp1d(F_inv, tau, kind="linear", bounds_error=False, fill_value=(0, 1))
    x_lower = np.linspace(miny, x, n_steps)
    x_upper = np.linspace(x, maxy, n_steps)
    f_lower = np.power(F(x_lower), 2)
    f_upper = np.power(1 - F(x_upper), 2)
    return np.trapz(f_lower, x=x_lower) + np.trapz(f_upper, x=x_upper) 
 
def logscore(tau, F_inv, x):
    pdf = np.diff(tau, prepend=0) / np.diff(F_inv, prepend=0)
    f = interpolate.interp1d(F_inv, pdf, kind="linear", bounds_error=False, fill_value=(np.min(pdf)/10, np.min(pdf)/10))
    xx = np.linspace(F_inv.max()*2, 10000)
    scale = np.trapz(f(xx), x=xx)
    return np.log(np.clip(f(x)/scale, 0, np.inf) + 1e-16)

def coverage(tau, F_inv, x, lo, hi, miny=0, maxy=100):
    tau = np.concatenate((np.array([0]), tau, np.array([1])))
    F_inv = np.concatenate((np.array([miny]), F_inv, np.array([maxy])))
    F_inv = interpolate.interp1d(tau, F_inv, kind="linear", bounds_error=False, fill_value=(miny,  maxy))
    return F_inv(lo) <= x <= F_inv(hi)


def load_dataset(dataset_path):
    dataset = load_datasets(
      os.path.join(dataset_path, "metadata"), 
      os.path.join(dataset_path, "train"), 
      os.path.join(dataset_path, "test")
    )
    return dataset

def make_quantiles(dataset, num_quantiles, quantile_boundary):
    targets = []
    for d in dataset.train:
        targets += d["target"].tolist()
    targets = np.array(targets)
    l = np.quantile(targets, quantile_boundary)
    u = np.quantile(targets, 1 - quantile_boundary)
    intervals = np.linspace(l, u, num_quantiles)
    quants = np.unique([np.mean(targets <= g) for g in intervals])
    return quants.tolist()

def make_trainer(config):
    trainer = Trainer(epochs=config["epochs"],
                      learning_rate=config["learning_rate"],
                      learning_rate_decay_factor=config["learning_rate_decay_factor"], 
                      weight_decay=config["weight_decay"],
                      ctx=mx.cpu())
    return trainer

def make_mqrnn_estimator(trainer, quantiles, monotonic, freq, prediction_length = 1):
    estimator = MQRNNEstimator(
        freq=freq, 
        prediction_length= prediction_length, 
        quantiles = quantiles,
        monotonic=monotonic,
        trainer=trainer)
    return estimator

def make_deepar_estimator(trainer, num_pieces, freq, prediction_length = 1):
    estimator = DeepAREstimator(
        freq=freq, 
        prediction_length= prediction_length, 
        distr_output=PiecewiseLinearOutput(num_pieces),
        trainer=trainer)
    return estimator

def make_predictor(estimator, dataset):      
    predictor = estimator.train(training_data=dataset.train)
    return predictor

def evaluate(predictor, dataset, quantiles, prediction_length = 1):
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test, 
        predictor=predictor, 
        num_samples=int(1000),
    ) 
    forecast_lst = list(forecast_it)
    ts_lst = list(ts_it)
    
    m1 = []
    m2 = []
    m3 = []
    m4 = []
    metrics = {}
    
    for obs, forecast in tqdm(zip(dataset.test, forecast_lst), total=len(dataset.test)):
        curr = 0
        try:
            if prediction_length > 1:
                pred_quants = np.sort(forecast.forecast_array[:,-1].squeeze()[:])
            else:
                pred_quants = np.sort(forecast.forecast_array.squeeze()[:])
        except:
            samps = forecast.samples
            pred_quants = np.quantile(samps, quantiles)
        upper = 1
        lower = -1
        m1.append(crps(quantiles, pred_quants + curr, obs["target"][-1] + curr, miny=lower, maxy=upper))
        m2.append(coverage(quantiles, pred_quants + curr, obs["target"][-1] + curr, 0.20, 0.80, miny=lower, maxy=upper))
        m3.append(coverage(quantiles, pred_quants + curr, obs["target"][-1] + curr, 0.10, 0.90, miny=lower, maxy=upper))
        m4.append(coverage(quantiles, pred_quants + curr, obs["target"][-1] + curr, 0.025, 0.975,  miny=lower, maxy=upper))
    metrics["crps"] = np.mean(m1)
    metrics["60cover"] = np.mean(m2)   
    metrics["80cover"] = np.mean(m3)   
    metrics["95cover"] = np.mean(m4)
    return metrics, forecast_lst, ts_lst

def save_predictor(predictor, model_path):
    os.makedirs(model_path, exist_ok=True)
    predictor.serialize(Path(model_path))


def main(args):

    with open(args.config_path) as json_file:
        config = json.load(json_file)
    dataset = load_dataset(args.dataset_path)
    quantiles = make_quantiles(dataset, config["num_quantiles"], args.quantile_boundary)
    trainer = make_trainer(config)

    if args.deepar:
        estimator = make_deepar_estimator(trainer, args.num_pieces, dataset.metadata.freq)
    else:
        estimator = make_mqrnn_estimator(trainer, quantiles, args.monotonic, dataset.metadata.freq)
    predictor = make_predictor(estimator, dataset)
    save_predictor(predictor, args.model_path)
    metrics, _, _ = evaluate(predictor, dataset, quantiles)

    print("All Test Data: ")
    print(metrics)
    if args.dataset_outlier_path is not "False":
        dataset_outliers = load_dataset(args.dataset_outlier_path)
        metrics_outliers, _, _ = evaluate(predictor, dataset_outliers, quantiles)
        print("Outlier Test Data: ")
        print(metrics_outliers)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('dataset_path', type=str,
                        help='Path to dataset')
    parser.add_argument('model_path', type=str,
                        help='Path to save model at')
    parser.add_argument('--config_path', type=str, default="configs/default.json",
                        help='Path where configs are stored.')
    parser.add_argument('--dataset_outlier_path', type=str, default="False",
                        help='Path to outlier data (optional).')
    parser.add_argument('--quantile_boundary', type=float, default=1e-3,
                        help='Smallest quantile to forecast')
    parser.add_argument('--pred_length', type=int, default=1,
                        help='Train monotonic model.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Training epochs')
    parser.add_argument('--deepar', type=bool, default=False,
                        help='Use DeepAR. ')
    parser.add_argument('--num_pieces', type=int, default=30,
                        help='Number of pieces for piecewise linear. ')
    parser.add_argument('--monotonic', type=bool, default=False,
                        help='Train monotonic model.')

        
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    print("Starting to run")
    main(args)
    
