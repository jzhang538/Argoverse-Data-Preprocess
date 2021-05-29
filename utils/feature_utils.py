#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Modified from https://github.com/xk-huang/yet-another-vectornet
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
from utils.config import color_dict
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from utils.object_utils import get_nearby_moving_obj_feature_ls
from utils.lane_utils import get_nearby_lane_feature_ls, get_halluc_lane, get_end_point_lane_id
from utils.agent_utils import get_agent_feature_ls
from utils.cand_utils import get_candidate_centerlines_for_trajectory
from utils.viz_utils import *
import pdb 
from utils.config import _MAX_SEARCH_RADIUS_CENTERLINES,_MAX_CENTERLINE_CANDIDATES


def compute_feature_for_one_seq(traj_df: pd.DataFrame, am: ArgoverseMap, obs_len: int = 20, lane_radius: int = 5, obj_radius: int = 10, viz: bool = False, mode='nearby', query_bbox=[-65, 65, -65, 65]) -> List[List]:
    """
    return lane & track features
    args:
        mode: 'rect' or 'nearby'
    returns:
        agent_feature_ls:
            list of target agent
        obj_feature_ls:
            list of (list of nearby agent feature)
        lane_feature_ls:
            list of (list of lane segment feature)
        norm_center np.ndarray: (2, )
    """
    # normalize timestamps
    traj_df['TIMESTAMP'] -= np.min(traj_df['TIMESTAMP'].values)
    seq_ts = np.unique(traj_df['TIMESTAMP'].values)

    city_name = traj_df['CITY_NAME'].iloc[0]
    agent_df = None
    agent_x_end, agent_y_end, start_x, start_y, query_x, query_y, norm_center = [
        None] * 7
    # agent traj & its start/end point
    for obj_type, remain_df in traj_df.groupby('OBJECT_TYPE'):
        # sorted already according to timestamp
        if obj_type == 'AGENT':
            agent_df = remain_df
            start_x, start_y = agent_df[['X', 'Y']].values[0]
            agent_x_end, agent_y_end = agent_df[['X', 'Y']].values[-1]
            query_x, query_y = agent_df[['X', 'Y']].values[obs_len-1]
            norm_center = np.array([query_x, query_y])
            break
        else:
            raise ValueError(f"cannot find 'agent' object type")

    # get agent features
    agent_feature = get_agent_feature_ls(agent_df, obs_len, norm_center)
    hist_xy = agent_feature[0]
    hist_len = np.sum(np.sqrt((hist_xy[1:,0]-hist_xy[:-1,0])**2 + (hist_xy[1:,1]-hist_xy[:-1,1])**2))
    
    # search lanes from the last observed point of agent
    lane_feature_ls = get_nearby_lane_feature_ls(
        am, agent_df, obs_len, city_name, lane_radius, norm_center, mode=mode, query_bbox=query_bbox)

    # search candidate centerlanes
    agent_xy_obs = agent_df[['X', 'Y']].values[:obs_len]
    agent_xy_fut = agent_df[['X', 'Y']].values[obs_len:]
    candidate_centerlines, candidate_lane_seqs = get_candidate_centerlines_for_trajectory(
                agent_xy_obs,
                agent_xy_fut,
                city_name,
                am,
                hist_len,
                norm_center,
                viz=False,
                max_search_radius=_MAX_SEARCH_RADIUS_CENTERLINES,
                seq_len=50,
                max_candidates=_MAX_CENTERLINE_CANDIDATES,
            )

    # search nearby moving objects from the last observed point of agent
    obj_feature_ls = get_nearby_moving_obj_feature_ls(
        agent_df, traj_df, obs_len, seq_ts, norm_center)

    # ep_lane_ids, ep_nearby_lane_ids = get_end_point_lane_id(am, city_name, [agent_x_end,agent_y_end])
    return [agent_feature, obj_feature_ls, lane_feature_ls, candidate_centerlines, candidate_lane_seqs, norm_center]


def trans_gt_offset_format(gt):
    """
    >Our predicted trajectories are parameterized as per-stepcoordinate offsets, starting from the last observed location.We rotate the coordinate system based on the heading of the target vehicle at the last observed location.
    """
    assert gt.shape == (30, 2) or gt.shape == (0, 2), f"{gt.shape} is wrong"
    # for test, no gt, just return a (0, 2) ndarray
    if gt.shape == (0, 2):
        return gt

    offset_gt = np.vstack((gt[0], gt[1:] - gt[:-1]))
    assert (offset_gt.cumsum(axis=0) -
            gt).sum() < 1e-6, f"{(offset_gt.cumsum(axis=0) -gt).sum()}"

    return offset_gt


def encoding_features(agent_feature, obj_feature_ls, lane_feature_ls, candidate_centerlines, candidate_lane_seqs):
    """
    args:
        agent_feature_ls:
            list of (xys, ts, agent_df['TRACK_ID'].iloc[0], gt_xys)
        obj_feature_ls:
            list of list of (xys, ts, mask, track_id, gt_xys, gt_mask)
        lane_feature_ls:
            list of list of lane a segment feature, centerline, lane_info1, lane_info2, lane_id
        candidate_centerlines:
            useless
        candidate_lane_seqs:
            useless 
    returns:
        pd.DataFrame
    """
    gt = agent_feature[-1]
    nbrs_nd = np.empty((0,4))
    nbrs_gt = np.empty((0,3))
    lane_nd = np.empty((0,7))
    lane_id2mask = {}
    lane_id2order = {}

    # encoding agent feature
    # xy,ts,mask,id
    agent_len = agent_feature[0].shape[0]
    agent_nd = np.hstack((agent_feature[0], agent_feature[1].reshape((-1, 1)), np.ones((agent_len, 1))))
    assert agent_nd.shape[1] == 4, "agent_traj feature dim 1 is not correct"

    if(len(obj_feature_ls)>0):
        # encoding obj feature
        # xy,ts,mask,id
        # gt: xy, mask
        for obj_feature in obj_feature_ls:
            obj_len = obj_feature[0].shape[0]
            obj_nd = np.hstack((obj_feature[0], obj_feature[1].reshape((-1, 1)), obj_feature[2].reshape((-1, 1))))
            assert obj_nd.shape[1] == 4, "obj_traj feature dim 1 is not correct"
            nbrs_nd = np.vstack([nbrs_nd,obj_nd])
            
            gt_len = obj_feature[4].shape[0]
            obj_gt =np.hstack((obj_feature[4], obj_feature[5].reshape((-1, 1))))
            assert obj_gt.shape[1] == 3, "obj_gt feature dim 1 is not correct"
            nbrs_gt = np.vstack([nbrs_gt,obj_gt])
        # nbrs_nd [nbrs_num,20,4]
        nbrs_nd = nbrs_nd.reshape([-1,20,4])
        # nbrs_gt [nbrs_num,30,3]
        nbrs_gt = nbrs_gt.reshape([-1,30,3])

    ct = 0
    pre_lane_len = lane_nd.shape[0]
    if(len(lane_feature_ls)>0):
        # encodeing lane feature
        # lane vector: point1_xy, point2_xy, is_intersection, turn_direction, is_traffic_control
        for lane_feature in lane_feature_ls:
            l_lane_len = lane_feature[0].shape[0]
            l_lane_nd = np.hstack((lane_feature[0], np.ones((l_lane_len, 1)) * lane_feature[1], np.ones((l_lane_len, 1)) * lane_feature[2], np.ones((l_lane_len, 1)) * lane_feature[3]))
            assert l_lane_nd.shape[1] == 7, "lane feature dim 1 is not correct"
            lane_nd = np.vstack([lane_nd,l_lane_nd])

            lane_id2mask[ct] = (pre_lane_len, lane_nd.shape[0])
            lane_id2order[lane_feature[4]] = ct 
            pre_lane_len = lane_nd.shape[0]
            ct+=1
    # lane [Vector_num,7]
    
    # add neighbor lanes
    if(len(lane_feature_ls)>0):
        nbr_lane_nd = []
        for lane_feature in lane_feature_ls:
            predecessor_lanes = lane_feature[5]
            successor_lanes = lane_feature[6]
            adjacent_lanes = lane_feature[7]
            
            p_ls = []
            if(predecessor_lanes!=None):
                for lane_id in predecessor_lanes:
                    if lane_id in lane_id2order:
                        nbr_id = lane_id2order[lane_id]
                        p_ls.append(nbr_id)
            s_ls = []
            if(successor_lanes!=None):
                for lane_id in successor_lanes:
                    if lane_id in lane_id2order:
                        nbr_id = lane_id2order[lane_id]
                        s_ls.append(nbr_id)
            adj_ls = []
            if(adjacent_lanes!=None):
                for lane_id in adjacent_lanes:
                    if lane_id!=None and lane_id in lane_id2order:
                        nbr_id = lane_id2order[lane_id]
                        adj_ls.append(nbr_id)

            nbr_lane_nd.append([p_ls, s_ls, adj_ls])

    # # ep_lane_ids, ep_nearby_lane_ids
    # ep_lane_orders = []
    # flag = False
    # for lane_id in ep_lane_ids:
    #     if lane_id in lane_id2order:
    #         lane_order = lane_id2order[lane_id]
    #         ep_lane_orders.append(lane_order)
    #         flag = True
    # ep_lane_orders=np.array(ep_lane_orders)
    # if(flag==False):
    #     print("end point not in the selected range")
    # print(ep_lane_orders)

    # ep_nearby_lane_orders = []
    # for lane_id in ep_nearby_lane_ids:
    #     if lane_id in lane_id2order:
    #         lane_order = lane_id2order[lane_id]
    #         ep_nearby_lane_orders.append(lane_order)
    # ep_nearby_lane_orders = np.array(ep_nearby_lane_orders)
    # print(ep_nearby_lane_orders)


    # ct = 0
    # cand_id2order_seq = {}
    # # candidate line mask
    # for i in range(len(candidate_lane_seqs)):
    #     candidate_lane_seq = candidate_lane_seqs[i]
    #     order_seq = []
    #     for lane_id in candidate_lane_seq:
    #         if lane_id not in lane_id2order:
    #             break
    #         lane_order = lane_id2order[lane_id]
    #         order_seq.append(lane_order)
    #     cand_id2order_seq[ct] = order_seq
    #     ct +=1

    # # candidate centerlines
    # cand_nd = np.empty((0,2))
    # cand_id2mask = {}
    # ct = 0
    # pre_cand_len = cand_nd.shape[0]
    # for i in range(len(candidate_centerlines)):
    #     candidate_centerline = candidate_centerlines[i]
    #     assert candidate_centerline.shape[1] == 2, "cand feature dim 1 is not correct"
    #     cand_nd = np.vstack([cand_nd,candidate_centerline])

    #     cand_id2mask[ct] = (pre_cand_len, cand_nd.shape[0])
    #     pre_cand_len = cand_nd.shape[0]
    #     ct+=1


    # transform gt to offset_gt
    offset_gt = trans_gt_offset_format(gt)
    
    # Now the features are:
    # In current version, we don't use candidate lanes.
    # data = [[agent_nd.astype(np.float32), offset_gt, nbrs_nd.astype(np.float32), nbrs_gt.astype(np.float32), lane_nd.astype(np.float32), lane_id2mask, \
    #     cand_id2order_seq, cand_nd, cand_id2mask]]

    # LANES_NBRS indicates predecessor_lanes, adjacent_lanes, successor_lanes respectively
    data = [[agent_nd.astype(np.float32), offset_gt.astype(np.float32), nbrs_nd.astype(np.float32), nbrs_gt.astype(np.float32), lane_nd.astype(np.float32), lane_id2mask, nbr_lane_nd]]
    return pd.DataFrame(
        data,
        columns=["AGENT_FEATURE", "GT", "NBRS_FEATURE", "NBRS_GT", "LANES_FEATURE", "LANES_MASK", "LANES_NBRS"]
    )


def save_features(df, name, dir_=None):
    if dir_ is None:
        dir_ = './input_data'
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    name = f"features_{name}.pkl"
    df.to_pickle(
        os.path.join(dir_, name)
    )
