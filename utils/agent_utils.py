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
import utils.config


def get_agent_feature_ls(agent_df, obs_len, norm_center):
    """
    args:
    returns: 
        list of (track, timetamp, track_id, gt_track)
    """
    xys, gt_xys = agent_df[["X", "Y"]].values[:obs_len], agent_df[[
        "X", "Y"]].values[obs_len:]
    xys -= norm_center  # normalize to last observed timestamp point of agent
    gt_xys -= norm_center  # normalize to last observed timestamp point of agent
    ts = agent_df['TIMESTAMP'].values[:obs_len]

    return [xys, ts, agent_df['TRACK_ID'].iloc[0], gt_xys]
