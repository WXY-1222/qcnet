# Copyright (c) 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
from __future__ import annotations

import os
import pickle
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


class InteractionDIGIRDataset(Dataset):
    """Load INTERACTION data preprocessed by DIGIR into QCNet-compatible HeteroData."""

    _VEHICLE_TYPE_TO_AV2_TYPE = {
        0: 0,  # car -> vehicle
        1: 4,  # truck/bus -> bus
        2: 1,  # pedestrian
        3: 3,  # bicycle -> cyclist
        4: 2,  # motorcycle -> motorcyclist
    }

    def __init__(
            self,
            data_path: str,
            split: str,
            transform: Optional[Callable] = None,
            num_historical_steps: int = 8,
            num_future_steps: int = 12,
            max_samples: Optional[int] = None,
            use_kg: bool = True) -> None:
        super(InteractionDIGIRDataset, self).__init__()
        self.data_path = os.path.expanduser(os.path.normpath(data_path))
        self.split = split
        self.transform = transform
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps
        self.max_samples = max_samples
        self.use_kg = use_kg

        with open(self.data_path, 'rb') as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise TypeError('interaction_digir pickle must be a dict payload')

        split_key = split
        if split_key not in payload:
            if split == 'test' and 'val' in payload:
                split_key = 'val'
            else:
                raise KeyError(f"Split '{split}' not found in {self.data_path}")

        split_samples = payload.get(split_key, [])
        if not isinstance(split_samples, list):
            raise TypeError(f"Split '{split_key}' in {self.data_path} must be a list")
        if max_samples is not None and max_samples > 0:
            split_samples = split_samples[:max_samples]
        self.samples: List[Dict[str, Any]] = split_samples
        self.sample_locations: List[str] = [str(s.get('location_name', 'UNKNOWN')) for s in self.samples]

        self.kg_default = payload.get('kg', None)
        self.kg_per_location = payload.get('kg_per_location', {}) or {}
        self.config = payload.get('config', {}) or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> HeteroData:
        data = self._sample_to_heterodata(self.samples[idx])
        if self.transform is not None:
            data = self.transform(data)
        return data

    def _sample_to_heterodata(self, sample: Dict[str, Any]) -> HeteroData:
        traj_np = np.asarray(sample['trajectory'], dtype=np.float32)
        fut_np = np.asarray(sample['future_trajectory'], dtype=np.float32)
        if traj_np.ndim != 3:
            raise ValueError("sample['trajectory'] must be (N,H,C)")
        if fut_np.ndim != 3:
            raise ValueError("sample['future_trajectory'] must be (N,F,C)")

        if traj_np.shape[1] < self.num_historical_steps:
            raise ValueError(
                f"Historical steps in sample ({traj_np.shape[1]}) < configured num_historical_steps "
                f"({self.num_historical_steps})")
        if fut_np.shape[1] < self.num_future_steps:
            raise ValueError(
                f"Future steps in sample ({fut_np.shape[1]}) < configured num_future_steps ({self.num_future_steps})")

        num_agents = min(traj_np.shape[0], fut_np.shape[0])
        if num_agents <= 0:
            raise ValueError('Sample has no valid agents')

        traj = torch.from_numpy(traj_np[:num_agents, :self.num_historical_steps]).float()
        future_xy = torch.from_numpy(fut_np[:num_agents, :self.num_future_steps, :2]).float()

        position = torch.cat([traj[..., :2], future_xy], dim=1)
        heading_hist = traj[..., 2]
        heading_future = self._infer_future_heading(future_xy, heading_hist[:, -1])
        heading = torch.cat([heading_hist, heading_future], dim=1)

        velocity_hist = self._build_hist_velocity(traj, heading_hist)
        velocity_future = self._build_future_velocity(traj[..., :2], future_xy)
        velocity = torch.cat([velocity_hist, velocity_future], dim=1)

        valid_mask = torch.ones(num_agents, self.num_steps, dtype=torch.bool)
        predict_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        predict_mask[:, self.num_historical_steps:] = True

        agent_type = self._build_agent_type(sample, num_agents)
        agent_category = torch.full((num_agents,), 3, dtype=torch.uint8)
        av_index = torch.tensor(0, dtype=torch.long)
        agent_ids = [f"agent_{i}" for i in range(num_agents)]

        map_features = self._build_map_features(sample)
        num_map_nodes = map_features['map_polygon']['num_nodes']
        point_to_polygon_edge_index = torch.stack(
            [torch.arange(num_map_nodes, dtype=torch.long),
             torch.arange(num_map_nodes, dtype=torch.long)],
            dim=0)

        case_id = sample.get('case_id', 'unknown_case')
        location_name = sample.get('location_name', 'INTERACTION')
        scenario_id = f"{location_name}_{case_id}_{sample.get('start_frame', 0)}"

        data = HeteroData()
        data['scenario_id'] = scenario_id
        data['city'] = location_name

        data['agent']['num_nodes'] = num_agents
        data['agent']['av_index'] = av_index
        data['agent']['valid_mask'] = valid_mask
        data['agent']['predict_mask'] = predict_mask
        data['agent']['id'] = agent_ids
        data['agent']['type'] = agent_type
        data['agent']['category'] = agent_category
        data['agent']['position'] = position
        data['agent']['heading'] = heading
        data['agent']['velocity'] = velocity

        data['map_polygon']['num_nodes'] = num_map_nodes
        data['map_polygon']['position'] = map_features['map_polygon']['position']
        data['map_polygon']['orientation'] = map_features['map_polygon']['orientation']
        data['map_polygon']['type'] = map_features['map_polygon']['type']
        data['map_polygon']['is_intersection'] = map_features['map_polygon']['is_intersection']

        data['map_point']['num_nodes'] = num_map_nodes
        data['map_point']['position'] = map_features['map_point']['position']
        data['map_point']['orientation'] = map_features['map_point']['orientation']
        data['map_point']['magnitude'] = map_features['map_point']['magnitude']
        data['map_point']['type'] = map_features['map_point']['type']
        data['map_point']['side'] = map_features['map_point']['side']

        data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index
        data['map_polygon', 'to', 'map_polygon']['edge_index'] = map_features['edge_index']
        data['map_polygon', 'to', 'map_polygon']['type'] = map_features['edge_type']
        return data

    @staticmethod
    def _build_hist_velocity(traj: torch.Tensor, heading_hist: torch.Tensor) -> torch.Tensor:
        hist_xy = traj[..., :2]
        if traj.size(-1) >= 4:
            speed_hist = traj[..., 3].clamp_min(0.0)
            vel_hist = torch.zeros_like(hist_xy)
            vel_hist[..., 0] = speed_hist * torch.cos(heading_hist)
            vel_hist[..., 1] = speed_hist * torch.sin(heading_hist)
            return vel_hist
        vel_hist = torch.zeros_like(hist_xy)
        vel_hist[:, 1:] = hist_xy[:, 1:] - hist_xy[:, :-1]
        return vel_hist

    @staticmethod
    def _build_future_velocity(hist_xy: torch.Tensor, future_xy: torch.Tensor) -> torch.Tensor:
        n, f, _ = future_xy.size()
        vel_future = future_xy.new_zeros((n, f, 2))
        vel_future[:, 0] = future_xy[:, 0] - hist_xy[:, -1]
        if f > 1:
            vel_future[:, 1:] = future_xy[:, 1:] - future_xy[:, :-1]
        return vel_future

    @staticmethod
    def _infer_future_heading(future_xy: torch.Tensor, last_hist_heading: torch.Tensor) -> torch.Tensor:
        n, f, _ = future_xy.size()
        heading_future = future_xy.new_zeros((n, f))
        if f == 0:
            return heading_future
        step_vec = future_xy.new_zeros((n, f, 2))
        if f >= 1:
            step_vec[:, 0] = 0.0
        if f > 1:
            step_vec[:, 1:] = future_xy[:, 1:] - future_xy[:, :-1]
        heading_future[:] = torch.atan2(step_vec[..., 1], step_vec[..., 0])
        heading_future[:, 0] = last_hist_heading
        return heading_future

    def _build_agent_type(self, sample: Dict[str, Any], num_agents: int) -> torch.Tensor:
        vtypes = sample.get('vehicle_types', None)
        if vtypes is None:
            return torch.zeros(num_agents, dtype=torch.uint8)
        raw = np.asarray(vtypes, dtype=np.int64)[:num_agents]
        mapped = np.zeros(num_agents, dtype=np.uint8)
        for i in range(raw.shape[0]):
            mapped[i] = np.uint8(self._VEHICLE_TYPE_TO_AV2_TYPE.get(int(raw[i]), 0))
        return torch.from_numpy(mapped)

    def _select_kg(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.use_kg:
            return None
        location_name = sample.get('location_name', None)
        if location_name is not None and location_name in self.kg_per_location:
            return self.kg_per_location[location_name]
        return self.kg_default

    def _build_map_features(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        kg = self._select_kg(sample)
        if kg is None:
            positions = np.zeros((1, 2), dtype=np.float32)
            facility_types = np.zeros((1,), dtype=np.int64)
            edge_index_np = np.zeros((2, 0), dtype=np.int64)
            edge_types_np = np.zeros((0,), dtype=np.int64)
        else:
            positions = np.asarray(kg.get('positions', np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
            facility_types = np.asarray(kg.get('facility_types', np.zeros((0,), dtype=np.int64)), dtype=np.int64)
            edge_index_np = np.asarray(kg.get('edge_index', np.zeros((2, 0), dtype=np.int64)), dtype=np.int64)
            edge_types_np = np.asarray(kg.get('edge_types', np.zeros((0,), dtype=np.int64)), dtype=np.int64)

            if positions.ndim != 2 or positions.shape[1] < 2:
                positions = np.zeros((0, 2), dtype=np.float32)
            if facility_types.ndim != 1:
                facility_types = np.zeros((0,), dtype=np.int64)
            if edge_index_np.ndim != 2 or edge_index_np.shape[0] != 2:
                edge_index_np = np.zeros((2, 0), dtype=np.int64)
            if edge_types_np.ndim != 1:
                edge_types_np = np.zeros((0,), dtype=np.int64)

        if positions.shape[0] == 0:
            positions = np.zeros((1, 2), dtype=np.float32)
            facility_types = np.zeros((1,), dtype=np.int64)
            edge_index_np = np.zeros((2, 0), dtype=np.int64)
            edge_types_np = np.zeros((0,), dtype=np.int64)

        num_nodes = positions.shape[0]
        if facility_types.shape[0] < num_nodes:
            padded = np.zeros((num_nodes,), dtype=np.int64)
            padded[:facility_types.shape[0]] = facility_types
            facility_types = padded
        else:
            facility_types = facility_types[:num_nodes]

        edge_index = torch.from_numpy(edge_index_np).long()
        if edge_index.numel() > 0:
            valid_edge = ((edge_index[0] >= 0) &
                          (edge_index[0] < num_nodes) &
                          (edge_index[1] >= 0) &
                          (edge_index[1] < num_nodes))
            edge_index = edge_index[:, valid_edge]
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        edge_type = torch.from_numpy(edge_types_np).long()
        if edge_type.numel() != edge_index.size(1):
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
        edge_type = torch.remainder(edge_type, 5).to(torch.uint8)

        map_pos = torch.from_numpy(positions[:, :2]).float()
        facility = torch.from_numpy(facility_types).long()
        map_orientation = self._build_map_orientation(map_pos, edge_index)

        point_position = map_pos.clone()
        point_orientation = map_orientation.clone()
        point_magnitude = map_pos.new_zeros((num_nodes,))
        point_type = torch.clamp(facility, min=0, max=16).to(torch.uint8)
        point_side = torch.zeros(num_nodes, dtype=torch.uint8)

        polygon_type = torch.clamp(facility, min=0, max=3).to(torch.uint8)
        polygon_is_intersection = torch.zeros(num_nodes, dtype=torch.uint8)

        return {
            'map_point': {
                'position': point_position,
                'orientation': point_orientation,
                'magnitude': point_magnitude,
                'type': point_type,
                'side': point_side,
            },
            'map_polygon': {
                'position': map_pos,
                'orientation': map_orientation,
                'type': polygon_type,
                'is_intersection': polygon_is_intersection,
            },
            'edge_index': edge_index,
            'edge_type': edge_type,
        }

    @staticmethod
    def _build_map_orientation(positions: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = positions.size(0)
        orientation = torch.zeros(num_nodes, dtype=torch.float32)
        if edge_index.numel() == 0:
            return orientation
        used = torch.zeros(num_nodes, dtype=torch.bool)
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for s, d in zip(src, dst):
            if s < 0 or s >= num_nodes or d < 0 or d >= num_nodes or used[s]:
                continue
            vec = positions[d] - positions[s]
            orientation[s] = torch.atan2(vec[1], vec[0])
            used[s] = True
        return orientation
