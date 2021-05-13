import pdb
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import (
    get_nt_distance,
    remove_overlapping_lane_seq,
)
from argoverse.utils.mpl_plotting_utils import visualize_centerline
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os

def get_point_in_polygon_score(lane_seq: List[int],
                               xy_seq: np.ndarray, city_name: str,
                               am: ArgoverseMap) -> int:
    """Get the number of coordinates that lie insde the lane seq polygon.

    Args:
        lane_seq: Sequence of lane ids
        xy_seq: Trajectory coordinates
        city_name: City name (PITT/MIA)
        am: Argoverse map_api instance
    Returns:
        point_in_polygon_score: Number of coordinates in the trajectory that lie within the lane sequence

    """
    lane_seq_polygon = cascaded_union([
        Polygon(am.get_lane_segment_polygon(lane, city_name)).buffer(0)
        for lane in lane_seq
    ])
    point_in_polygon_score = 0
    for xy in xy_seq:
        point_in_polygon_score += lane_seq_polygon.contains(Point(xy))
    return point_in_polygon_score

def sort_lanes_based_on_point_in_polygon_score(
            lane_seqs: List[List[int]],
            xy_seq: np.ndarray,
            city_name: str,
            am: ArgoverseMap,
    ) -> List[List[int]]:
        """Filter lane_seqs based on the number of coordinates inside the bounding polygon of lanes.

        Args:
            lane_seqs: Sequence of lane sequences
            xy_seq: Trajectory coordinates
            city_name: City name (PITT/MIA)
            am: Argoverse map_api instance
        Returns:
            sorted_lane_seqs: Sequences of lane sequences sorted based on the point_in_polygon score

        """
        point_in_polygon_scores = []
        for lane_seq in lane_seqs:
            point_in_polygon_scores.append(
                get_point_in_polygon_score(lane_seq, xy_seq, city_name,
                                                am))
        randomized_tiebreaker = np.random.random(len(point_in_polygon_scores))
        sorted_point_in_polygon_scores_idx = np.lexsort(
            (randomized_tiebreaker, np.array(point_in_polygon_scores)))[::-1]
        sorted_lane_seqs = [
            lane_seqs[i] for i in sorted_point_in_polygon_scores_idx
        ]
        sorted_scores = [
            point_in_polygon_scores[i]
            for i in sorted_point_in_polygon_scores_idx
        ]
        return sorted_lane_seqs, sorted_scores


def get_heuristic_centerlines_for_test_set(
            lane_seqs: List[List[int]],
            xy_seq: np.ndarray,
            city_name: str,
            am: ArgoverseMap,
            max_candidates: int,
            scores: List[int],
    ) -> List[np.ndarray]:
        """Sort based on distance along centerline and return the centerlines.
        
        Args:
            lane_seqs: Sequence of lane sequences
            xy_seq: Trajectory coordinates
            city_name: City name (PITT/MIA)
            am: Argoverse map_api instance
            max_candidates: Maximum number of centerlines to return
        Return:
            sorted_candidate_centerlines: Centerlines in the order of their score 

        """
        alinged_lane_seqs = []
        aligned_centerlines = []
        diverse_lane_seqs = []
        diverse_centerlines = []
        diverse_scores = []
        num_candidates = 0

        # Get first half as aligned centerlines
        aligned_cl_count = 0
        for i in range(len(lane_seqs)):
            lane_seq = lane_seqs[i]
            score = scores[i]
            diverse = True
            centerline = am.get_cl_from_lane_seq([lane_seq], city_name)[0]
            if aligned_cl_count < int(max_candidates / 2):
                start_dist = LineString(centerline).project(Point(xy_seq[0]))
                end_dist = LineString(centerline).project(Point(xy_seq[-1]))
                if end_dist > start_dist:
                    aligned_cl_count += 1
                    aligned_centerlines.append(centerline)
                    alinged_lane_seqs.append(lane_seq)
                    diverse = False
            if diverse:
                diverse_centerlines.append(centerline)
                diverse_lane_seqs.append(lane_seq)
                diverse_scores.append(score)

        # print(len(diverse_centerlines))
        num_diverse_centerlines = min(len(diverse_centerlines),
                                      max_candidates - aligned_cl_count)
        test_centerlines = aligned_centerlines
        test_lane_seqs = alinged_lane_seqs
        if num_diverse_centerlines > 0:
            probabilities = ([
                float(score + 1) / (sum(diverse_scores) + len(diverse_scores))
                for score in diverse_scores
            ] if sum(diverse_scores) > 0 else [1.0 / len(diverse_scores)] *
                             len(diverse_scores))
            diverse_centerlines_idx = np.random.choice(
                range(len(probabilities)),
                num_diverse_centerlines,
                replace=False,
                p=probabilities,
            )
            diverse_centerlines = [
                diverse_centerlines[i] for i in diverse_centerlines_idx
            ]
            diverse_lane_seqs = [
                diverse_lane_seqs[i] for i in diverse_centerlines_idx
            ]
            test_centerlines += diverse_centerlines
            test_lane_seqs += diverse_lane_seqs

        return test_centerlines, test_lane_seqs

def dfs(
        am,
        lane_id: int,
        city_name: str,
        dist: float = 0,
        threshold: float = 30,
        extend_along_predecessor: bool = False,
    ) -> List[List[int]]:
        """
        Perform depth first search over lane graph up to the threshold.

        Args:
            lane_id: Starting lane_id (Eg. 12345)
            city_name
            dist: Distance of the current path
            threshold: Threshold after which to stop the search
            extend_along_predecessor: if true, dfs over predecessors, else successors

        Returns:
            lanes_to_return (list of list of integers): List of sequence of lane ids
                Eg. [[12345, 12346, 12347], [12345, 12348]]

        """
        if dist > threshold:
            return [[lane_id]]
        else:
            traversed_lanes = []
            child_lanes = (
                am.get_lane_segment_predecessor_ids(lane_id, city_name)
                if extend_along_predecessor
                else am.get_lane_segment_successor_ids(lane_id, city_name)
            )
            if child_lanes is not None:
                for child in child_lanes:
                    centerline = am.get_lane_segment_centerline(child, city_name)
                    cl_length = LineString(centerline).length
                    curr_lane_ids = am.dfs(child, city_name, dist + cl_length, threshold, extend_along_predecessor)
                    traversed_lanes.extend(curr_lane_ids)
            if len(traversed_lanes) == 0:
                return [[lane_id]]
            lanes_to_return = []
            for lane_seq in traversed_lanes:
                lanes_to_return.append(lane_seq + [lane_id] if extend_along_predecessor else [lane_id] + lane_seq)
            return lanes_to_return

def get_candidate_centerlines_for_trajectory(
            xy_obs: np.ndarray,
            xy_fut: np.ndarray,
            city_name: str,
            am: ArgoverseMap,
            hist_len: int,
            norm_center: np.ndarray,
            viz: bool = False,
            max_search_radius: float = 50.0,
            seq_len: int = 50,
            max_candidates: int = 10,
    ) -> List[np.ndarray]:
        """Get centerline candidates upto a threshold.

        Algorithm:
        1. Take the lanes in the bubble of last observed coordinate
        2. Extend before and after considering all possible candidates
        3. Get centerlines based on point in polygon score.

        Args:
            xy_obs: Trajectory coordinates, 
            xy_fut: Same
            city_name: City name, 
            am: Argoverse map_api instance, 
            viz: Visualize candidate centerlines, 
            max_search_radius: Max search radius for finding nearby lanes in meters,
            seq_len: Sequence length, 
            max_candidates: Maximum number of centerlines to return, 
            mode: train/val/test mode

        Returns:
            candidate_centerlines: List of candidate centerlines

        """
        _MANHATTAN_THRESHOLD = 4.0  # meters
        dfs_threshold_front = np.max((hist_len*2,40))
        # dfs_threshold_back = hist_len

        # Get all lane candidates within a bubble
        curr_lane_candidates = am.get_lane_ids_in_xy_bbox(
            xy_obs[-1, 0], xy_obs[-1, 1], city_name, _MANHATTAN_THRESHOLD)
        # Keep expanding the bubble until at least 1 lane is found
        while (len(curr_lane_candidates) < 1
               and _MANHATTAN_THRESHOLD < max_search_radius):
            _MANHATTAN_THRESHOLD *= 2
            curr_lane_candidates = am.get_lane_ids_in_xy_bbox(
                xy_obs[-1, 0], xy_obs[-1, 1], city_name, _MANHATTAN_THRESHOLD)
        # assert len(curr_lane_candidates) > 0, "No nearby lanes found!!"

        # DFS to get all successor and predecessor candidates
        obs_pred_lanes: List[Sequence[int]] = []
        for lane in curr_lane_candidates:
            candidates_future = dfs(am,lane, city_name, 0,
                                        dfs_threshold_front)
            for future_lane_seq in candidates_future:
                obs_pred_lanes.append(future_lane_seq)
            # candidates_past = dfs(am,lane, city_name, 0, dfs_threshold_back,
            #                           True)

            # # Merge past and future
            # for past_lane_seq in candidates_past:
            #     for future_lane_seq in candidates_future:
            #         assert (
            #             past_lane_seq[-1] == future_lane_seq[0]
            #         ), "Incorrect DFS for candidate lanes past and future"
            #         obs_pred_lanes.append(past_lane_seq + future_lane_seq[1:])

        # Removing overlapping lanes
        obs_pred_lanes = remove_overlapping_lane_seq(obs_pred_lanes)

        # # Sort lanes based on point in polygon score
        # obs_pred_lanes, scores = sort_lanes_based_on_point_in_polygon_score(
        #     obs_pred_lanes, xy_obs, city_name, am)
        # # If the best centerline is not along the direction of travel, re-sort
        # candidate_centerlines = get_heuristic_centerlines_for_test_set(
        #         obs_pred_lanes, xy_obs, city_name, am, max_candidates, scores)

        if(xy_fut.shape[0]!=0):
            xy = np.vstack([xy_obs,xy_fut])
            obs_pred_lanes, scores = sort_lanes_based_on_point_in_polygon_score(
                obs_pred_lanes, xy, city_name, am)
            candidate_centerlines, candidate_lane_seqs = get_heuristic_centerlines_for_test_set(
                obs_pred_lanes, xy, city_name, am, max_candidates, scores)
        else:
            obs_pred_lanes, scores = sort_lanes_based_on_point_in_polygon_score(
                obs_pred_lanes, xy_obs, city_name, am)
            candidate_centerlines, candidate_lane_seqs = get_heuristic_centerlines_for_test_set(
                obs_pred_lanes, xy_obs, city_name, am, max_candidates, scores)
        
        for i in range(len(candidate_centerlines)):
            candidate_centerlines[i] -= norm_center
        # centerline = am.get_cl_from_lane_seq([lane_seq], city_name)[0]
        if viz:
            xy_obs -= norm_center
            plt.figure(0, figsize=(8, 7))
            for centerline_coords in candidate_centerlines:
                visualize_centerline(centerline_coords)
            # for lane_seq in candidate_lane_seqs:
            #     centerline_coords = am.get_cl_from_lane_seq([lane_seq], city_name)[0]
            #     visualize_centerline(centerline_coords)
            plt.plot(
                xy_obs[:, 0],
                xy_obs[:, 1],
                "-",
                color="#d33e4c",
                alpha=1,
                linewidth=3,
                zorder=15,
            )

            final_x = xy_obs[-1, 0]
            final_y = xy_obs[-1, 1]

            plt.plot(
                final_x,
                final_y,
                "o",
                color="#d33e4c",
                alpha=1,
                markersize=10,
                zorder=15,
            )
            plt.xlabel("Map X")
            plt.ylabel("Map Y")
            plt.axis("off")
            plt.title(f"Number of candidates = {len(candidate_centerlines)}")
            plt.show()

        return candidate_centerlines, candidate_lane_seqs