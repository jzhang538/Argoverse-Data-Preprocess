#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Modified from https://github.com/xk-huang/yet-another-vectornet
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com

RAW_DATA_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
}
LANE_WIDTH = {'MIA': 3.84, 'PIT': 3.97}
VELOCITY_THRESHOLD = 0.0
# Number of timesteps the track should exist to be considered in social context
EXIST_THRESHOLD = (5)
# index of the sorted velocity to look at, to call it as stationary
STATIONARY_THRESHOLD = (13)
color_dict = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"}

LANE_RADIUS = 100 # nearby lanes
OBJ_RADIUS = 90 # nearby objects
OBS_LEN = 20
# nearby candidate centerlines
_MAX_SEARCH_RADIUS_CENTERLINES = 50.0
_MAX_CENTERLINE_CANDIDATES = 10

DATA_DIR = '/home/SENSETIME/zhangjinghuai1/Desktop/argo-raw-data-test'
INTERMEDIATE_DATA_DIR = './interm_data'

VIS=True
