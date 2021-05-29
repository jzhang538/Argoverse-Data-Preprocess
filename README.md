#  Code for Data Preprocessing of Argoverse Dataset
-> Also official data preprocessing for paper ["Multimodal Motion Prediction with Stacked Transformers. (CVPR 2021)"](https://github.com/decisionforce/mmTransformer)

## Usage

0) Install [Argoverse-api](https://github.com/argoai/argoverse-api/tree/master/argoverse). Download `HD-maps` in argoverse-api as instructed.

1) Prepare raw Argoverse dataset:
    
    Put all data (folders named `train/val/test` or a single folder `sample`) in `data` folder.
    
    An example folder structure:
    ```
    data - train - *.csv
         \        \ ...
          \
           \- val - *.csv
            \       \ ...
             \
              \- test - *.csv
                       \ ...
    ```
2) Modify the config file `utils/config.py`. Use the proper env paths and arguments.

3) Feature preprocessing, save intermediate data input features (compute_feature_module.py)
    ```
    $ python compute_feature_module.py
    ```
## Result Intermediate Data (Per Case)

"AGENT_FEATURE": [20, 5] # x, y, timestamp, mask, agent_id

"GT": [30,2] # x, y 

"NBRS_FEATURE": [num_neighbors, 20, 5] # x, y, timestamp, mask, agent_id

"NBRS_GT": [num_neighbors, 30, 3] # x, y, mask

"LANES_FEATURE": [num_vectors, 7] # point1_xy, point2_xy, is_intersection, turn_direction, is_traffic_control, 

"LANES_MASK": [num_lanes, 2] # each lane_id -> start and end index of lane vectors in LANES_FEATURE

"LANES_NBRS": [list of predecessor ids, list of sucessor ids, list of adjacent ids] # id -> lane_id in LANES_MASK
