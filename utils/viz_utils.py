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
import os
from utils.config import color_dict
import torch
import random

def visualize_vectors(df):
    agent = df["AGENT_FEATURE"].values[0]
    agent_gt = df["GT"].values[0]
    nbrs = df["NBRS_FEATURE"].values[0]
    nbrs_gt = df["NBRS_GT"].values[0]
    lanes = df["LANES_FEATURE"].values[0]
    lanes_mask = df["LANES_MASK"].values[0]

    # order_seqs = df["ORDER_SEQS"].values[0]
    # cands = df["CAND_FEATURE"].values[0]
    # cands_mask = df["CAND_MASK"].values[0]
    # ep_lane_order = df["EP_LANE_ORDER"].values[0]
    # ep_nearby_lane_order = df["EP_NEARBY_LANE_ORDER"].values[0]

    plt.figure(figsize=(15,15))
    plt.xlim(xmin=-100,xmax=100)
    plt.ylim(ymin=-100,ymax=100)

    for i in range(len(lanes_mask)):
        polyline = lanes[lanes_mask[i][0]:lanes_mask[i][1]]
        coords=np.vstack([polyline[:,:2],polyline[-1,2:4]])
        # different turn directions
        if(polyline[0][5]==2):
            a, = plt.plot(coords[:, 0], coords[:, 1], '--', color='black', linewidth=1)
        elif(polyline[0][5]==3):
            a, = plt.plot(coords[:, 0], coords[:, 1], '--', color='black', linewidth=1)
        else:
            a, = plt.plot(coords[:, 0], coords[:, 1], '--', color='black', linewidth=1)

    # Draw adjacent lanes
    lanes_nbrs = df["LANES_NBRS"].values[0]
    select = random.randint(0,len(lanes_nbrs)-1)
    lane_nbrs = lanes_nbrs[select]
    # Highlight the selected lane segment
    polyline = lanes[lanes_mask[select][0]:lanes_mask[select][1]]
    coords=np.vstack([polyline[:,:2],polyline[-1,2:4]])
    print("Self length", len(coords))
    o, = plt.plot(coords[:, 0], coords[:, 1], '--', color='orange', linewidth=2)
    # p
    for lane_nbr_id in lane_nbrs[0]:
        polyline = lanes[lanes_mask[lane_nbr_id][0]:lanes_mask[lane_nbr_id][1]]
        coords=np.vstack([polyline[:,:2],polyline[-1,2:4]])
        o, = plt.plot(coords[:, 0], coords[:, 1], '--', color='blue', linewidth=2)
    # s
    for lane_nbr_id in lane_nbrs[1]:
        polyline = lanes[lanes_mask[lane_nbr_id][0]:lanes_mask[lane_nbr_id][1]]
        coords=np.vstack([polyline[:,:2],polyline[-1,2:4]])
        o, = plt.plot(coords[:, 0], coords[:, 1], '--', color='purple', linewidth=2)
    # adj
    for lane_nbr_id in lane_nbrs[2]:
        polyline = lanes[lanes_mask[lane_nbr_id][0]:lanes_mask[lane_nbr_id][1]]
        coords=np.vstack([polyline[:,:2],polyline[-1,2:4]])
        o, = plt.plot(coords[:, 0], coords[:, 1], '--', color='yellow', linewidth=2)
    
    # agent related 
    polyline = agent
    coords = polyline[:,:2]
    b, = plt.plot(coords[:, 0], coords[:, 1], '-', color='green', linewidth=3, zorder=2)

    # offset -> trajectory coordinates
    coords = np.zeros((len(agent_gt)+1,2))
    for i in range(len(agent_gt)):
        temp=coords[i] + agent_gt[i]
        coords[i+1]=temp
    c, =plt.plot(coords[1:,0],coords[1:,1], '-', color='blue', linewidth=3, zorder=2)

    print("Number of nearby agents:", len(nbrs))
    if(len(nbrs)!=0):
        for i in range(len(nbrs)):
            polyline = nbrs[i]
            coords = polyline[:,:2]
            mask = polyline[:,3]
            coords = coords[np.where(mask==1)]
            ep = coords[-1]
            # print(coords)
            # print(mask)
            d, = plt.plot(coords[:, 0], coords[:, 1], '-', color='red', linewidth=1, zorder=2)

            polyline = nbrs_gt[i]
            coords = polyline[:,:2]
            mask = polyline[:,2]
            coords = coords[np.where(mask==1)]
            # print(coords)
            # print(mask)
            coords = np.vstack([ep,coords])
            e, = plt.plot(coords[:, 0], coords[:, 1], '-', color='pink', linewidth=1, zorder=2)

            # plt.scatter(coords[-1, 0], coords[-1, 1], color='purple', s=10, zorder=2)

    plt.axis('off')
    plt.show()