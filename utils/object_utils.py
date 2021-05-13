#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Modified from https://github.com/xk-huang/yet-another-vectornet
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from utils.config import VELOCITY_THRESHOLD, RAW_DATA_FORMAT, OBJ_RADIUS, EXIST_THRESHOLD


def compute_velocity(track_df: pd.DataFrame) -> List[float]:
    """Compute velocities for the given track.
    Args:
        track_df (pandas Dataframe): Data for the track
    Returns:
        vel (list of float): Velocity at each timestep
    """
    x_coord = track_df["X"].values
    y_coord = track_df["Y"].values
    timestamp = track_df["TIMESTAMP"].values

    vel_x, vel_y = zip(*[(
        (x_coord[i] - x_coord[i - 1]) /
        (float(timestamp[i]) - float(timestamp[i - 1])),
        (y_coord[i] - y_coord[i - 1]) /
        (float(timestamp[i]) - float(timestamp[i - 1])),
    ) for i in range(1, len(timestamp))])
    vel = [np.sqrt(x**2 + y**2) for x, y in zip(vel_x, vel_y)]
    return vel


def get_is_track_stationary(track_df: pd.DataFrame) -> bool:
    """Check if the track is stationary.
    Args:
        track_df (pandas Dataframe): Data for the track
    Return:
        _ (bool): True if track is stationary, else False

    """
    vel = compute_velocity(track_df)
    sorted_vel = sorted(vel)
    threshold_vel = sorted_vel[int(len(vel) / 2)]
    return True if threshold_vel < VELOCITY_THRESHOLD else False

def pad_track(
        track_df: pd.DataFrame,
        seq_timestamps: np.ndarray,
        base: int,
        track_len: int,
        raw_data_format: Dict[str, int],
) -> np.ndarray:
    """Pad incomplete tracks.
    Args:
        track_df (Dataframe): Dataframe for the track
        seq_timestamps (numpy array): All timestamps in the sequence
        base: base frame id (0 for observed trajectory, 20 for future trajectory)
        track_len (int): Length of whole trajectory (observed + future)
        raw_data_format (Dict): Format of the sequence
    Returns:
        padded_track_array (numpy array)
    """
    track_vals = track_df.values
    track_timestamps = track_df["TIMESTAMP"].values
    seq_timestamps = seq_timestamps[base:base+track_len]

    # start and index of the track in the sequence
    start_idx = np.where(seq_timestamps == track_timestamps[0])[0][0]
    end_idx = np.where(seq_timestamps == track_timestamps[-1])[0][0]

    # Edge padding in front and rear, i.e., repeat the first and last coordinates
    # if self.PADDING_TYPE == "REPEAT"
    padded_track_array = np.pad(track_vals,
                                ((start_idx, track_len - end_idx - 1),
                                    (0, 0)), "edge")

    mask = np.ones((end_idx+1-start_idx))
    mask = np.pad(mask,(start_idx, track_len - end_idx - 1),'constant')
    if padded_track_array.shape[0] < track_len:
        # rare case, just ignore
        return None, None, False

    # Overwrite the timestamps in padded part
    for i in range(padded_track_array.shape[0]):
        padded_track_array[i, 0] = seq_timestamps[i]
    assert mask.shape[0] == padded_track_array.shape[0]
    return padded_track_array,mask,True


# Important: We delete some invalid nearby agents according to our criterion!!! Refer to continue option
def get_nearby_moving_obj_feature_ls(agent_df, traj_df, obs_len, seq_ts, norm_center, fut_len=30):
    """
    args:
    returns: list of list, (track, timestamp, mask, track_id, gt_track, gt_mask)
    """
    obj_feature_ls = []
    query_x, query_y = agent_df[['X', 'Y']].values[obs_len-1]
    p0 = np.array([query_x, query_y])
    for track_id, remain_df in traj_df.groupby('TRACK_ID'):
        if remain_df['OBJECT_TYPE'].iloc[0] == 'AGENT':
            continue

        hist_df = remain_df[remain_df['TIMESTAMP'] <=
                   agent_df['TIMESTAMP'].values[obs_len-1]]
        if(len(hist_df)==0):
            continue
        # pad hist
        xys, ts = None, None
        if len(hist_df) < obs_len:
            paded_nd,mask,flag = pad_track(hist_df, seq_ts, 0, obs_len, RAW_DATA_FORMAT)
            if flag==False:
                continue
            xys = np.array(paded_nd[:, 3:5], dtype=np.float64)
            ts = np.array(paded_nd[:, 0], dtype=np.float64)
        else:
            xys = hist_df[['X', 'Y']].values
            ts = hist_df["TIMESTAMP"].values
            mask = np.ones((obs_len))
        
        p1 = xys[-1]
        if mask[-1]==0 or np.linalg.norm(p0 - p1) > OBJ_RADIUS:
            continue
        if(sum(mask)<=3):
            continue
        
        fut_df = remain_df[remain_df['TIMESTAMP'] >
                   agent_df['TIMESTAMP'].values[obs_len-1]]
        # pad future
        gt_xys = None
        if len(fut_df) == 0:
            gt_xys = np.zeros((fut_len,2))+norm_center
            gt_mask = np.zeros((fut_len))
        elif len(fut_df) < fut_len:
            paded_nd,gt_mask,flag = pad_track(fut_df, seq_ts, obs_len, fut_len, RAW_DATA_FORMAT)
            if flag==False:
                continue
            gt_xys = np.array(paded_nd[:, 3:5], dtype=np.float64)
        else:
            gt_xys = fut_df[['X', 'Y']].values
            gt_mask = np.ones((fut_len))

        xys -= norm_center  # normalize to last observed timestamp point of agent
        gt_xys -= norm_center

        assert xys.shape[0]==obs_len
        assert mask.shape[0]==obs_len
        assert gt_xys.shape[0]==fut_len
        assert gt_mask.shape[0]==fut_len
        obj_feature_ls.append(
            [xys, ts, mask, track_id, gt_xys, gt_mask])
    return obj_feature_ls
