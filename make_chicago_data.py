
import datetime
import json
import os

import joblib
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import pandas as pd
from gluonts.dataset import common
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from mxnet import gluon
from tqdm.notebook import tqdm

from preprocessing import make_event_study, save_gluon, train_test_split

np.random.seed(1000)

SOURCE_PATH = "/mnt/spacious/shared/vadim_all_sensors.csv"

def get_dataframe(source_path):
    df = pd.read_csv(source_path)
    df["ds"] = pd.to_datetime(df["ds"])
    df["target"] = df.groupby("DeviceId").Occupancy.fillna(method="ffill", limit=5)
    df["sensor"], _ = pd.factorize(df["DeviceId"])
    return df

def make_segments(df, diffed=True, drift_thresh=3600000000000.0*4, min_target_length=288//4):
    segments = []
    item_id = 0
    for s in df.sensor.unique():
        target_sequence = (df.loc[df.sensor == s, "target"]).to_numpy()
        target_sequence -= np.min(target_sequence)
        target_sequence /= np.max(target_sequence)
        ds_sequence =  (df.loc[df.sensor == s, "ds"]).to_numpy()[1:]
        start_idx = 0
        end_idx = 0
        diff_seq = np.diff(ds_sequence).astype(int)
        m = np.median(diff_seq)
        while True:  
            if not len(ds_sequence[start_idx:]):
                break
            try:
                diff_seq = np.diff(ds_sequence[start_idx:]).astype(int)
                delta_seq = np.cumsum(diff_seq - m)
                end_idx = np.argmax(np.abs(delta_seq) > drift_thresh) + start_idx
            except ValueError:
                break
            
            t_seq = target_sequence[start_idx:end_idx].astype(np.float32)
            d_seq = np.diff(target_sequence[start_idx:end_idx].astype(np.float32), prepend=0)
                
            seg = dict(
                start=ds_sequence[start_idx], 
                end=ds_sequence[end_idx],
                target=d_seq if diffed else t_seq,
                feat_static_cat=np.array([s], dtype=int),
                item_id = item_id,
                real_time = ds_sequence[:end_idx],
            )
            if diffed:
                seg["feat_dynamic_real"]= t_seq[np.newaxis, :]
            segments.append(seg)
            start_idx = end_idx + 1
            if start_idx >= target_sequence.shape[0]:
                break
            if np.all(np.isnan(target_sequence[end_idx:])):
                break
        item_id += 1
        segments = [s for s in segments if len(s["target"]) > min_target_length]
    return segments, item_id

def make_datasets():
    dates = [
        (pd.Timestamp(year=2012, month=1, day=1, freq="5min"), pd.Timestamp(year=2050, month=1, day=1, freq="5min")), # train-test.
        (pd.Timestamp(year=2010, month=1, day=1, freq="5min"), pd.Timestamp(year=2010, month=6, day=1, freq="5min")), # cv's 
        (pd.Timestamp(year=2010, month=6, day=1, freq="5min"), pd.Timestamp(year=2010, month=12, day=1, freq="5min")),
        (pd.Timestamp(year=2011, month=1, day=1, freq="5min"), pd.Timestamp(year=2011, month=6, day=1, freq="5min")),
        (pd.Timestamp(year=2011, month=6, day=1, freq="5min"), pd.Timestamp(year=2011, month=12, day=1, freq="5min")),
    ]

    print("Loading dataframe")
    df = get_dataframe(SOURCE_PATH)
    for diffed in [True, False]:
        print("Making segments")
        subname = "diffed" if diffed else "nondiffed"
        dataset_name = "chic_d" if diffed else "chic"
        segments, base_item_id = make_segments(df, diffed=diffed)
        os.makedirs(f"data/cv/{dataset_name}/", exist_ok=True)
        for i, (start_date, end_date) in enumerate(dates):
            print(start_date, end_date)
            item_id = base_item_id
            train_segments, test_segments = train_test_split(segments, start_date, end_date, item_id)
            print(len(train_segments), len(test_segments))
            if i > 0:
                save_path = f"data/cv/{dataset_name}/{dataset_name}_{i}.gluonts"
            else:
                save_path = f"data/{dataset_name}.gluonts"
            save_gluon(save_path, train_segments, test_segments, "5T")

        os.makedirs(f"data/examples/{dataset_name}/", exist_ok=True)
        print("making event studies")
        make_event_study(df, diffed,
            pd.date_range(start='2013-3-13', end='2013-3-14', freq = 's'), 
            15,
            f"data/examples/{dataset_name}/{dataset_name}_example_2013_3_13.pkl", 
            f"data/examples/{dataset_name}/{dataset_name}_reference_2013_3_13.pkl",
            "5T"
        )
        make_event_study(df, diffed,
            pd.date_range(start='2013-3-15', end='2013-3-16', freq = 's'), 
            15,
            f"data/examples/{dataset_name}/{dataset_name}_example_2013_3_15.pkl", 
            f"data/examples/{dataset_name}/{dataset_name}_reference_2013_3_15.pkl",
            "5T"
        )
        make_event_study(df, diffed,
            pd.date_range(start='2013-10-10', end='2013-10-11', freq = 's'), 
            15,
            f"data/examples/{dataset_name}/{dataset_name}_example_2013_10_10.pkl", 
            f"data/examples/{dataset_name}/{dataset_name}_reference_2013_10_10.pkl",
            "5T"
        )
    

if __name__ == "__main__":
    make_datasets()
