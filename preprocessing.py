
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

def make_event_study(df, diffed, time_range, sensor, save_path, reference_save_path, freq, downsamp=False, min_target_length=288//4):
    df["ds_TZ"] = df["ds"] - pd.Timedelta("5 hour")
    df["Time_of_Day"] = ((df.ds_TZ - df.ds_TZ.dt.normalize()) / pd.Timedelta('5 minute')).astype(int)
    start = time_range[0]
    end = time_range[-1]
    segments = []
    target_sequence = (df.loc[df.sensor == sensor, "target"]).to_numpy()
    df["target_standard"] = (df.target - np.nanmin(target_sequence)) / np.nanmax(target_sequence)
    target_sequence -= np.nanmin(target_sequence)
    target_sequence /= np.nanmax(target_sequence)
    if downsamp:
        target_sequence = target_sequence[::5]
    diff_sequence = np.diff(target_sequence, prepend=0)
    ds_sequence =  (df.loc[df.sensor == sensor, "ds"]).to_numpy()
    ds_TZ_sequence =  (df.loc[df.sensor == sensor, "ds_TZ"]).to_numpy()
    if downsamp:
        ds_sequence = ds_sequence[::5]
        ds_TZ_sequence = ds_TZ_sequence[::5]
    start_idx = np.argmax(ds_TZ_sequence >= start)
    end_idx = np.argmax(ds_TZ_sequence >= end)
    segments = []
    for idx in range(start_idx, end_idx):
        seg = dict(
            target = diff_sequence[(start_idx - min_target_length):idx] if diffed else target_sequence[(start_idx - min_target_length):idx],
            start = ds_sequence[start_idx - min_target_length],
            item_id = idx,
            feat_static_cat = np.array([sensor], dtype=int),
        )
        if diffed:
            seg["feat_dynamic_real"] = target_sequence[np.newaxis, (start_idx - min_target_length):idx]
        segments.append(seg)
    
    if "feat_dynamic_real" in seg.keys():
        analysis_dataset = ListDataset(data_iter=[{FieldName.TARGET: seg["target"],
                                FieldName.START: seg["start"], 
                                FieldName.FEAT_DYNAMIC_REAL: seg["feat_dynamic_real"],
                                FieldName.FEAT_STATIC_CAT: seg["feat_static_cat"], 
                                FieldName.ITEM_ID : seg["item_id"]}
                                for seg in segments if seg["target"].shape[0] > min_target_length],
                            freq=freq)
    else:
        analysis_dataset = ListDataset(data_iter=[{FieldName.TARGET: seg["target"],
                        FieldName.START: seg["start"], 
                        FieldName.FEAT_STATIC_CAT: seg["feat_static_cat"], 
                        FieldName.ITEM_ID : seg["item_id"]}
                        for seg in segments if seg["target"].shape[0] > min_target_length],
                    freq=freq)
    
    joblib.dump(analysis_dataset, save_path )
    joblib.dump(df.loc[(df.ds.dt.dayofweek == start.dayofweek) &(df.sensor == sensor), :].groupby("Time_of_Day").target_standard.median(), 
                reference_save_path)
def split_dataset(segments, thresh, item_id, min_target_length=288//4):
    lo_segments = []
    hi_segments = []
    for item in segments:
        start = item["start"]
        end = item["end"]
        if start <= thresh <= end:
            dates = item["real_time"]
            cut_idx = [idx for idx, d in enumerate(dates) if d >= thresh][0]
            lo_seg = dict(
                start=item["start"], 
                end=dates[cut_idx],
                target=item["target"][:cut_idx],
                feat_static_cat=item["feat_static_cat"],
                item_id =item["item_id"],
                real_time=item["real_time"][:cut_idx]
            )
            hi_seg = dict(
                start=dates[cut_idx],
                end=item["end"],
                target=item["target"][cut_idx:],
                feat_static_cat=item["feat_static_cat"],
                item_id = item["item_id"],
                real_time=item["real_time"][:cut_idx]
            )
            try:
                lo_seg["feat_dynamic_real"] = item["feat_dynamic_real"][:, :cut_idx]
                hi_seg["feat_dynamic_real"] = item["feat_dynamic_real"][:, cut_idx:]
            except KeyError:
                pass
            item_id += 1
            lo_segments.append(lo_seg)
            hi_segments.append(hi_seg)
        elif end <= thresh:
            lo_segments.append(item)
        else:
            hi_segments.append(item)
    lo_segments = [s for s in lo_segments if len(s["target"]) > min_target_length]
    hi_segments = [s for s in hi_segments if len(s["target"]) > min_target_length]
    return lo_segments, hi_segments

def train_test_split(segments, start_test, end_test, item_id):
    train_segments, hi_segments = split_dataset(segments, start_test, item_id)
    test_segments, _ = split_dataset(hi_segments, end_test, item_id)
    return train_segments, test_segments

    
def save_gluon(path, train_segments, test_segments, freq):
    if "feat_dynamic_real" in train_segments[0].keys():
        train_dataset = ListDataset(data_iter=[{FieldName.TARGET: seg["target"],
                                FieldName.START: seg["start"], 
                                FieldName.FEAT_DYNAMIC_REAL: seg["feat_dynamic_real"],
                                FieldName.FEAT_STATIC_CAT: seg["feat_static_cat"],  
                                FieldName.ITEM_ID : seg["item_id"]}
                                for seg in train_segments],
                            freq=freq)
        test_dataset = ListDataset(data_iter=[{FieldName.TARGET: seg["target"],
                                FieldName.START: seg["start"], 
                                FieldName.FEAT_DYNAMIC_REAL: seg["feat_dynamic_real"],
                                FieldName.FEAT_STATIC_CAT: seg["feat_static_cat"], 
                                FieldName.ITEM_ID : seg["item_id"]}
                                for seg in test_segments],
                            freq=freq)
    else:
        train_dataset = ListDataset(data_iter=[{FieldName.TARGET: seg["target"],
                                FieldName.START: seg["start"], 
                                FieldName.FEAT_STATIC_CAT: seg["feat_static_cat"],  
                                FieldName.ITEM_ID : seg["item_id"]}
                                for seg in train_segments],
                            freq=freq)
        test_dataset = ListDataset(data_iter=[{FieldName.TARGET: seg["target"],
                                FieldName.START: seg["start"], 
                                FieldName.FEAT_STATIC_CAT: seg["feat_static_cat"], 
                                FieldName.ITEM_ID : seg["item_id"]}
                                for seg in test_segments],
                            freq=freq)        

    metadata = common.MetaData(freq=freq, prediction_length=1)
    dataset = common.TrainDatasets(metadata, train_dataset, test_dataset)
    dataset.save(path)
