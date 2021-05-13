#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Modified from https://github.com/xk-huang/yet-another-vectornet
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
# %%
from utils.feature_utils import compute_feature_for_one_seq, encoding_features, save_features
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from utils.config import DATA_DIR, LANE_RADIUS, OBJ_RADIUS, OBS_LEN, INTERMEDIATE_DATA_DIR, VIS
from tqdm import tqdm
import re
import pickle
from utils.viz_utils import visualize_vectors

if __name__ == "__main__":
    am = ArgoverseMap()

    if not os.path.exists(INTERMEDIATE_DATA_DIR):
        os.makedirs(INTERMEDIATE_DATA_DIR)

    # Process each folder (train, val, test)
    for folder in os.listdir(DATA_DIR):
        afl = ArgoverseForecastingLoader(os.path.join(DATA_DIR, folder))
        norm_center_dict = {}
        for name in tqdm(afl.seq_list):
            afl_ = afl.get(name)
            path, name = os.path.split(name)
            name, ext = os.path.splitext(name)

            agent_feature, obj_feature_ls, lane_feature_ls, candidate_centerlines, candidate_lane_seqs, norm_center = compute_feature_for_one_seq(
                afl_.seq_df, am, OBS_LEN, LANE_RADIUS, OBJ_RADIUS, viz=False, mode='nearby')
            df = encoding_features(
                agent_feature, obj_feature_ls, lane_feature_ls, candidate_centerlines, candidate_lane_seqs)

            if VIS:
                visualize_vectors(df)
            save_features(df, name, os.path.join(
                INTERMEDIATE_DATA_DIR, f"{folder}_intermediate"))
            norm_center_dict[name] = norm_center
        with open(os.path.join(INTERMEDIATE_DATA_DIR, f"{folder}-norm_center_dict.pkl"), 'wb') as f:
            pickle.dump(norm_center_dict, f, pickle.HIGHEST_PROTOCOL)

# %%
