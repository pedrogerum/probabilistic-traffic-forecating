from glob import glob
import joblib
import numpy as np
from gluonts.dataset import common
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation.backtest import make_evaluation_predictions
from tqdm.notebook import tqdm
from pathlib import Path
from gluonts.model.predictor import Predictor
from quantile_forecast import load_dataset, make_quantiles
import matplotlib.pyplot as plt

def load_examples(source_path, reference_path):
    analysis_dataset = joblib.load(source_path)
    analysis_reference = joblib.load(reference_path)
    return analysis_dataset, analysis_reference

def get_predictions(predictor, dataset):
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor, 
    ) 

    pred_quants = []
    actual = []
    for obs, forecast in tqdm(zip(dataset, forecast_it), total=len(dataset)):
        curr = obs["feat_dynamic_real"][0, -1]
        pred_quants.append(curr + forecast.forecast_array.squeeze())
        actual.append(curr + obs["target"][-1])

    pred_quants = np.array(pred_quants)
    actual = np.array(actual)
    return pred_quants, actual

def plot_examples(pred_quants, actual, quantiles, reference, save_path):
    upper = pred_quants[:, np.argmin(np.abs(quantiles - 0.10))]
    lower = pred_quants[:,  np.argmin(np.abs(quantiles - 0.90))]
    upper_wide = pred_quants[:, np.argmin(np.abs(quantiles - 0.025))]
    lower_wide = pred_quants[:, np.argmin(np.abs(quantiles - 0.975))]
    x_reference = reference.index.to_numpy()
    x_reference = (x_reference / x_reference.max()) * 24 
    x = np.linspace(0, 24, upper.shape[0])
    median = pred_quants[:, pred_quants.shape[1]//2]

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, actual, "ko", markersize=1.5, label="Ordinary")
    ax.plot(x, median, color="blue", linewidth=1, alpha=0.5, label="Median")
    ax.fill_between(x, lower_wide, upper_wide, alpha=0.1, color="blue", label="95% PI")
    ax.fill_between(x, lower, upper, alpha=0.2, color="blue", label="80% PI")
    ax.plot(x_reference, reference, color="green", linestyle="dashed", label="Typical Median")
    ax.legend(loc="upper left")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Occupancy")
    ax.set_xlim(0, 24)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)

def main():
    from glob import glob
    dataset = load_dataset("data/chicago_diffed.gluonts")
    quantiles = np.array(make_quantiles(dataset, 100, 1e-3))
    predictor = Predictor.deserialize(Path("models/chicago-nonmono-diffed-short"))
    examples = sorted(list(glob("data/examples/chicago/diffed/chicago_diffed_example_*.pkl")))
    references = sorted(list(glob("data/examples/chicago/diffed/chicago_diffed_reference_*.pkl")))
    for src_path, ref_path in zip(examples, references):
        analysis_dataset, reference = load_examples(src_path, ref_path)
        pred_quants, actual = get_predictions(predictor, analysis_dataset)
        save_path = f"graphics/{Path(src_path).stem}.png"
        plot_examples(pred_quants, actual, quantiles, reference, save_path)



if __name__ == "__main__":
    main()