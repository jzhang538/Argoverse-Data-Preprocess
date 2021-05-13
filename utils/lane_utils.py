#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Modified from https://github.com/xk-huang/yet-another-vectornet
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
import pdb
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from utils.config import LANE_WIDTH

# Func: Two options to get right&left edge lanes
def get_halluc_lane(centerlane, city_name):
    """
    return left & right lane based on centerline
    args:
    returns:
        doubled_left_halluc_lane, doubled_right_halluc_lane, shaped in (N-1, 3)
    """
    if centerlane.shape[0] <= 1:
        raise ValueError('shape of centerlane error.')

    half_width = LANE_WIDTH[city_name] / 2
    rotate_quat = np.array([[0.0, -1.0], [1.0, 0.0]])
    halluc_lane_1, halluc_lane_2 = np.empty(
        (0, centerlane.shape[1]*2)), np.empty((0, centerlane.shape[1]*2))
    for i in range(centerlane.shape[0]-1):
        st, en = centerlane[i][:2], centerlane[i+1][:2]
        dx = en - st
        norm = np.linalg.norm(dx)
        e1, e2 = rotate_quat @ dx / norm, rotate_quat.T @ dx / norm
        lane_1 = np.hstack(
            (st + e1 * half_width, centerlane[i][2], en + e1 * half_width, centerlane[i+1][2]))
        lane_2 = np.hstack(
            (st + e2 * half_width, centerlane[i][2], en + e2 * half_width, centerlane[i+1][2]))
        halluc_lane_1 = np.vstack((halluc_lane_1, lane_1))
        halluc_lane_2 = np.vstack((halluc_lane_2, lane_2))

    #print(halluc_lane_1.shape,halluc_lane_2.shape)
    return halluc_lane_1, halluc_lane_2

def get_edge_lane(lane_polyline):
    lane_polyline = lane_polyline[:-1]
    
    break_point = int(len(lane_polyline)/2)
    edge1 = lane_polyline[:break_point,:2]
    edge2 = lane_polyline[-1:-break_point-1:-1,:2]
    # for i in range(break_point):
    #     plt.scatter(edge1[i,0],edge1[i,1],linewidths=i)
    #     plt.scatter(edge2[i,0],edge2[i,1],linewidths=i)
    # plt.show()
    return edge1,edge2

# Only support nearby in our implementation
def get_nearby_lane_feature_ls(am, agent_df, obs_len, city_name, lane_radius, norm_center, has_attr=False, mode='nearby', query_bbox=None):
    '''
    compute lane features
    args:
        norm_center: np.ndarray
        mode: 'nearby' return nearby lanes within the radius; 'rect' return lanes within the query bbox
        **kwargs: query_bbox= List[int, int, int, int]
    returns:
        list of list of lane a segment feature, formatted in [centerline, is_intersection, turn_direction, is_traffic_control, lane_id,
         predecessor_lanes, successor_lanes, adjacent_lanes]
    '''
    lane_feature_ls = []
    # print(mode)
    if mode == 'nearby':
        query_x, query_y = agent_df[['X', 'Y']].values[obs_len-1]
        nearby_lane_ids = am.get_lane_ids_in_xy_bbox(
            query_x, query_y, city_name, lane_radius)

        for lane_id in nearby_lane_ids:
            lane_cl = am.get_lane_segment_centerline(lane_id,city_name)
            is_intersection = am.lane_is_in_intersection(lane_id, city_name)
            turn_direction = am.get_lane_turn_direction(lane_id, city_name)
            traffic_control = am.lane_has_traffic_control_measure(lane_id, city_name)

            predecessor_lanes = am.get_lane_segment_predecessor_ids(lane_id,city_name)
            adjacent_lanes = am.get_lane_segment_adjacent_ids(lane_id,city_name)
            successor_lanes = am.get_lane_segment_successor_ids(lane_id,city_name)
            # print(predecessor_lanes, adjacent_lanes, successor_lanes)
            
            lane_info1 = 1
            if(is_intersection):
                lane_info1 = 2
            lane_info2 = 1
            if(turn_direction=="LEFT"):
                lane_info2 = 2
            elif(turn_direction=="RIGHT"):
                lane_info2 = 3
            lane_info3 = 1
            if(traffic_control):
                lane_info3 = 2

            centerline = lane_cl[:,:2]
            centerline -= norm_center
            centerline = np.hstack([centerline[:-1],centerline[1:]])

            lane_feature_ls.append(
                [centerline, lane_info1, lane_info2, lane_info3, lane_id, predecessor_lanes, successor_lanes, adjacent_lanes])
    else:
        raise ValueError(f"{mode} is not in {'nearby'}")

    return lane_feature_ls

# Func: get IDs of nearby lane segments where end point locates
def get_end_point_lane_id(am, city_name, end_point, max_search_radius=10.0):
    end_point_lane_ids = am.get_lane_segments_containing_xy(end_point[0],end_point[1],city_name)

    # Multiple
    if(len(end_point_lane_ids)>1):
        print(len(end_point_lane_ids))
    elif(len(end_point_lane_ids)==0):
        # find nearest

        _MANHATTAN_THRESHOLD = 2.5 
        nearby_lane_ids = am.get_lane_ids_in_xy_bbox(
            end_point[0], end_point[1], city_name, _MANHATTAN_THRESHOLD)
        # Keep expanding the bubble until at least 1 lane is found
        while(len(nearby_lane_ids) < 1 and _MANHATTAN_THRESHOLD <= max_search_radius):
            _MANHATTAN_THRESHOLD *= 2
            nearby_lane_ids = am.get_lane_ids_in_xy_bbox(
                end_point[0], end_point[1], city_name, _MANHATTAN_THRESHOLD)
        assert len(nearby_lane_ids) > 0, "No nearby lanes found!!"

        if(len(nearby_lane_ids)>1):
            ct = 0
            nearest_idx = -1
            mindis2poly=100
            for lane_id in nearby_lane_ids:
                lane_poly = am.get_lane_segment_polygon(lane_id,city_name)[:,:2]
                dis = np.sqrt((lane_poly[:,0]-end_point[0])**2 + (lane_poly[:,1]-end_point[1])**2)
                dis2poly = min(dis)
                if(mindis2poly>dis2poly):
                    nearest_idx = ct
                    mindis2poly = dis2poly
                ct+=1

            end_point_lane_ids = [nearby_lane_ids[nearest_idx]]
        else:
            end_point_lane_ids = nearby_lane_ids

    ep_nearby_lane_ids = []
    for lane_id in end_point_lane_ids:
        predecessor_lanes = am.get_lane_segment_predecessor_ids(lane_id,city_name)
        adjacent_lanes = am.get_lane_segment_adjacent_ids(lane_id,city_name)
        successor_lanes = am.get_lane_segment_successor_ids(lane_id,city_name)
        
        for lane_id in predecessor_lanes:
            if(lane_id!=None):
                lane_poly = am.get_lane_segment_polygon(lane_id,city_name)[:,:2]
                dis = np.sqrt((lane_poly[:,0]-end_point[0])**2 + (lane_poly[:,1]-end_point[1])**2)
                dis2poly = min(dis)
                if(dis2poly<=2.0):
                    ep_nearby_lane_ids.append(lane_id)

        for lane_id in adjacent_lanes:
            if(lane_id!=None):
                lane_poly = am.get_lane_segment_polygon(lane_id,city_name)[:,:2]
                dis = np.sqrt((lane_poly[:,0]-end_point[0])**2 + (lane_poly[:,1]-end_point[1])**2)
                dis2poly = min(dis)
                if(dis2poly<=1.0):
                    ep_nearby_lane_ids.append(lane_id)

        for lane_id in successor_lanes:
            if(lane_id!=None):
                lane_poly = am.get_lane_segment_polygon(lane_id,city_name)[:,:2]
                dis = np.sqrt((lane_poly[:,0]-end_point[0])**2 + (lane_poly[:,1]-end_point[1])**2)
                dis2poly = min(dis)
                if(dis2poly<=2.0):
                    ep_nearby_lane_ids.append(lane_id)

    return end_point_lane_ids, ep_nearby_lane_ids