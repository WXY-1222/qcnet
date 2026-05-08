# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from layers import MLPLayer
from modules.qcnet_decoder import BidirectionalMambaBlock
from utils import weight_init


class TopoSSMDecoder(nn.Module):
    """
    Topology-first decoder that keeps the QCNet scene encoder and replaces the
    QCNet decoder with:
      topology query / goal proposal -> explicit corridor extraction ->
      spatial Bi-Mamba over the corridor -> temporal Bi-Mamba rollout ->
      topology-aware scoring.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 topo_ssm_layers: int,
                 topo_mamba_d_state: int,
                 topo_mamba_d_conv: int,
                 topo_mamba_expand: int,
                 dropout: float,
                 topo_proposal_type: str = 'goal_mlp',
                 topo_goal_distance_weight: float = 0.05,
                 topo_goal_residual_scale: float = 0.25,
                 topo_goal_anchor_blend: float = 1.0,
                 topo_mode_endpoint_scale: float = 0.08,
                 topo_anchor_basis_scale: float = 0.20,
                 topo_polyline_control_scale: float = 0.12,
                 topo_route_slot_longitudinal_scale: float = 0.20,
                 topo_route_slot_lateral_scale: float = 0.12,
                 topo_route_slot_topk: int = 12,
                 topo_route_slot_soft_temperature: float = 0.35,
                 topo_aux_score: bool = False,
                 topo_aux_score_detach: bool = True,
                 corridor_dist_norm: float = 50.0) -> None:
        super(TopoSSMDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.topo_proposal_type = topo_proposal_type
        self.topo_goal_distance_weight = topo_goal_distance_weight
        self.topo_goal_residual_scale = topo_goal_residual_scale
        self.topo_goal_anchor_blend = topo_goal_anchor_blend
        self.topo_mode_endpoint_scale = topo_mode_endpoint_scale
        self.topo_anchor_basis_scale = topo_anchor_basis_scale
        self.topo_polyline_control_scale = topo_polyline_control_scale
        self.topo_route_slot_longitudinal_scale = topo_route_slot_longitudinal_scale
        self.topo_route_slot_lateral_scale = topo_route_slot_lateral_scale
        self.topo_route_slot_topk = topo_route_slot_topk
        self.topo_route_slot_soft_temperature = topo_route_slot_soft_temperature
        self.topo_aux_score = topo_aux_score
        self.topo_aux_score_detach = topo_aux_score_detach
        self.corridor_dist_norm = corridor_dist_norm
        if topo_proposal_type not in (
                'goal_mlp', 'mode_endpoint', 'corridor_mode_endpoint', 'corridor_goal', 'corridor_residual',
                'corridor_query', 'corridor_query_safe', 'decomp_endpoint', 'decomp_endpoint_polyline',
                'mode_endpoint_anchorbasis', 'mode_endpoint_polyline_readout', 'mode_endpoint_polyline_lite',
                'corridor_multi_anchor', 'route_slot_polyline', 'soft_route_slot_polyline',
                'interaction_decomp_endpoint', 'interaction_cv_endpoint', 'attn_endpoint', 'topo_query',
                'lane_prior_anchor'):
            raise ValueError(f'{topo_proposal_type} is not a valid topo_proposal_type')
        if topo_proposal_type in (
                'decomp_endpoint', 'decomp_endpoint_polyline', 'mode_endpoint_anchorbasis',
                'mode_endpoint_polyline_readout', 'mode_endpoint_polyline_lite', 'corridor_multi_anchor',
                'route_slot_polyline', 'soft_route_slot_polyline', 'interaction_decomp_endpoint',
                'interaction_cv_endpoint', 'attn_endpoint', 'topo_query', 'lane_prior_anchor') and output_dim != 2:
            raise ValueError(f'{topo_proposal_type} currently requires output_dim == 2')

        self.mode_emb = nn.Embedding(num_modes, hidden_dim)
        self.query_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.to_goal = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        if topo_proposal_type in (
                'mode_endpoint', 'corridor_mode_endpoint', 'mode_endpoint_anchorbasis',
                'mode_endpoint_polyline_readout', 'mode_endpoint_polyline_lite', 'corridor_multi_anchor',
                'route_slot_polyline', 'soft_route_slot_polyline'):
            self.mode_endpoint_anchor = nn.Parameter(torch.zeros(num_modes, output_dim))
            self.to_mode_endpoint_delta = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2 + output_dim * 2),
                nn.Linear(hidden_dim * 2 + output_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        if topo_proposal_type == 'mode_endpoint_anchorbasis':
            self.mode_anchor_basis_offset = nn.Parameter(torch.zeros(num_modes, num_future_steps, 2))
        if topo_proposal_type == 'mode_endpoint_polyline_readout':
            self.num_polyline_control_points = 2
            self.polyline_control_anchor = nn.Parameter(torch.zeros(num_modes, self.num_polyline_control_points, 2))
            self.to_readout_goal_delta = nn.Sequential(
                nn.LayerNorm(hidden_dim * 4 + output_dim * 2 + 2),
                nn.Linear(hidden_dim * 4 + output_dim * 2 + 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            self.to_readout_polyline_control = nn.Sequential(
                nn.LayerNorm(hidden_dim * 4 + output_dim * 2 + 2 + self.num_polyline_control_points * 2),
                nn.Linear(hidden_dim * 4 + output_dim * 2 + 2 + self.num_polyline_control_points * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_polyline_control_points * 2),
            )
        if topo_proposal_type == 'mode_endpoint_polyline_lite':
            self.num_polyline_control_points = 1
            self.polyline_control_anchor = nn.Parameter(torch.zeros(num_modes, self.num_polyline_control_points, 2))
            self.to_polyline_control_lite = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2 + output_dim + self.num_polyline_control_points * 2),
                nn.Linear(hidden_dim * 2 + output_dim + self.num_polyline_control_points * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_polyline_control_points * 2),
            )
        if topo_proposal_type == 'corridor_multi_anchor':
            self.num_polyline_control_points = 1
            self.polyline_control_anchor = nn.Parameter(torch.zeros(num_modes, self.num_polyline_control_points, 2))
            self.to_corridor_multi_goal_delta = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3 + output_dim * 2),
                nn.Linear(hidden_dim * 3 + output_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            self.to_corridor_multi_control = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3 + output_dim * 3 + self.num_polyline_control_points * 2),
                nn.Linear(hidden_dim * 3 + output_dim * 3 + self.num_polyline_control_points * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_polyline_control_points * 2),
            )
        if topo_proposal_type in ('route_slot_polyline', 'soft_route_slot_polyline'):
            self.num_polyline_control_points = 1
            self.route_slot_axis_anchor = nn.Parameter(torch.zeros(num_modes, 2))
            self.polyline_control_anchor = nn.Parameter(torch.zeros(num_modes, self.num_polyline_control_points, 2))
            self.to_route_slot_axis = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3 + output_dim * 2 + 2),
                nn.Linear(hidden_dim * 3 + output_dim * 2 + 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
            self.to_route_slot_control = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3 + output_dim * 2 + 2 + self.num_polyline_control_points * 2),
                nn.Linear(hidden_dim * 3 + output_dim * 2 + 2 + self.num_polyline_control_points * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_polyline_control_points * 2),
            )
        if topo_proposal_type == 'lane_prior_anchor':
            self.lane_prior_axis_anchor = nn.Parameter(torch.zeros(num_modes, 2))
            self.to_lane_prior_axis = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3 + output_dim * 3),
                nn.Linear(hidden_dim * 3 + output_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
        if topo_proposal_type in ('decomp_endpoint', 'decomp_endpoint_polyline'):
            self.endpoint_axis_anchor = nn.Parameter(torch.zeros(num_modes, 2))
            self.to_endpoint_axis = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2 + output_dim + 2),
                nn.Linear(hidden_dim * 2 + output_dim + 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
        if topo_proposal_type in ('interaction_decomp_endpoint', 'interaction_cv_endpoint'):
            self.interaction_axis_anchor = nn.Parameter(torch.zeros(num_modes, 2))
            self.to_interaction_context = nn.Sequential(
                nn.LayerNorm(hidden_dim + 3),
                nn.Linear(hidden_dim + 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.to_interaction_axis = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3 + output_dim + 2),
                nn.Linear(hidden_dim * 3 + output_dim + 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
        if topo_proposal_type == 'attn_endpoint':
            self.attn_endpoint_anchor = nn.Parameter(torch.zeros(num_modes, output_dim))
            self.attn_token_geo_proj = nn.Sequential(
                nn.LayerNorm(3),
                nn.Linear(3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.attn_q = nn.Linear(hidden_dim, hidden_dim)
            self.attn_k = nn.Linear(hidden_dim, hidden_dim)
            self.attn_v = nn.Linear(hidden_dim, hidden_dim)
            self.attn_out = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.to_attn_endpoint_delta = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3 + output_dim * 2),
                nn.Linear(hidden_dim * 3 + output_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        if topo_proposal_type == 'topo_query':
            self.topo_query_endpoint_anchor = nn.Parameter(torch.zeros(num_modes, output_dim))
            self.topo_query_agent_geo_proj = nn.Sequential(
                nn.LayerNorm(3),
                nn.Linear(3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.topo_query_map_geo_proj = nn.Sequential(
                nn.LayerNorm(3),
                nn.Linear(3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.topo_query_agent_q = nn.Linear(hidden_dim, hidden_dim)
            self.topo_query_agent_k = nn.Linear(hidden_dim, hidden_dim)
            self.topo_query_agent_v = nn.Linear(hidden_dim, hidden_dim)
            self.topo_query_map_q = nn.Linear(hidden_dim, hidden_dim)
            self.topo_query_map_k = nn.Linear(hidden_dim, hidden_dim)
            self.topo_query_map_v = nn.Linear(hidden_dim, hidden_dim)
            self.topo_query_context_fuse = nn.Sequential(
                nn.LayerNorm(hidden_dim * 4 + output_dim * 3),
                nn.Linear(hidden_dim * 4 + output_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.to_topo_query_endpoint_delta = nn.Sequential(
                nn.LayerNorm(hidden_dim + output_dim * 2),
                nn.Linear(hidden_dim + output_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        if topo_proposal_type == 'decomp_endpoint_polyline':
            self.num_polyline_control_points = 2
            self.polyline_control_anchor = nn.Parameter(torch.zeros(num_modes, self.num_polyline_control_points, 2))
            self.to_polyline_control = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2 + output_dim + 2 + self.num_polyline_control_points * 2),
                nn.Linear(hidden_dim * 2 + output_dim + 2 + self.num_polyline_control_points * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_polyline_control_points * 2),
            )
        if topo_proposal_type == 'corridor_mode_endpoint':
            self.to_corridor_mode_endpoint_delta = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2 + output_dim * 2),
                nn.Linear(hidden_dim * 2 + output_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        if topo_proposal_type in ('corridor_residual', 'corridor_query', 'corridor_query_safe'):
            self.to_corridor_goal_delta = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2 + output_dim),
                nn.Linear(hidden_dim * 2 + output_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        if topo_proposal_type in ('corridor_query', 'corridor_query_safe'):
            self.to_corridor_anchor_delta = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2 + output_dim + 2),
                nn.Linear(hidden_dim * 2 + output_dim + 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_future_steps * output_dim),
            )
        self.to_anchor_residual = MLPLayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_future_steps * output_dim,
        )
        if topo_proposal_type == 'decomp_endpoint_polyline':
            for param in self.to_anchor_residual.parameters():
                param.requires_grad_(False)

        token_dim = hidden_dim * 3 + output_dim * 2 + 2
        self.corridor_token_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.spatial_ssm = nn.ModuleList([
            BidirectionalMambaBlock(hidden_dim, topo_mamba_d_state, topo_mamba_d_conv, topo_mamba_expand, dropout)
            for _ in range(topo_ssm_layers)
        ])
        self.to_loc_propose_pos = MLPLayer(hidden_dim, hidden_dim, output_dim)
        self.to_scale_propose_pos = MLPLayer(hidden_dim, hidden_dim, output_dim)

        rollout_dim = hidden_dim + output_dim + output_dim + 1
        self.rollout_token_proj = nn.Sequential(
            nn.LayerNorm(rollout_dim),
            nn.Linear(rollout_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_ssm = nn.ModuleList([
            BidirectionalMambaBlock(hidden_dim, topo_mamba_d_state, topo_mamba_d_conv, topo_mamba_expand, dropout)
            for _ in range(topo_ssm_layers)
        ])
        self.to_loc_refine_pos = MLPLayer(hidden_dim, hidden_dim, output_dim)
        self.to_scale_refine_pos = MLPLayer(hidden_dim, hidden_dim, output_dim)
        self.to_pi = nn.Sequential(
            nn.LayerNorm(hidden_dim + 2),
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if topo_aux_score:
            self.to_topo_aux_pi = nn.Sequential(
                nn.LayerNorm(hidden_dim + 4),
                nn.Linear(hidden_dim + 4, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        self.apply(weight_init)
        if topo_proposal_type in (
                'mode_endpoint', 'corridor_mode_endpoint', 'mode_endpoint_anchorbasis',
                'mode_endpoint_polyline_readout', 'mode_endpoint_polyline_lite', 'corridor_multi_anchor',
                'route_slot_polyline', 'soft_route_slot_polyline'):
            self._init_mode_endpoint_anchor()
            nn.init.zeros_(self.to_mode_endpoint_delta[-1].weight)
            nn.init.zeros_(self.to_mode_endpoint_delta[-1].bias)
        if topo_proposal_type == 'mode_endpoint_anchorbasis':
            nn.init.zeros_(self.mode_anchor_basis_offset)
        if topo_proposal_type == 'mode_endpoint_polyline_readout':
            self._init_readout_polyline_anchor()
            nn.init.zeros_(self.to_readout_goal_delta[-1].weight)
            nn.init.zeros_(self.to_readout_goal_delta[-1].bias)
            nn.init.zeros_(self.to_readout_polyline_control[-1].weight)
            nn.init.zeros_(self.to_readout_polyline_control[-1].bias)
        if topo_proposal_type == 'mode_endpoint_polyline_lite':
            self._init_lite_polyline_anchor()
            nn.init.zeros_(self.to_polyline_control_lite[-1].weight)
            nn.init.zeros_(self.to_polyline_control_lite[-1].bias)
        if topo_proposal_type == 'corridor_multi_anchor':
            self._init_lite_polyline_anchor()
            nn.init.zeros_(self.to_corridor_multi_goal_delta[-1].weight)
            nn.init.zeros_(self.to_corridor_multi_goal_delta[-1].bias)
            nn.init.zeros_(self.to_corridor_multi_control[-1].weight)
            nn.init.zeros_(self.to_corridor_multi_control[-1].bias)
        if topo_proposal_type in ('route_slot_polyline', 'soft_route_slot_polyline'):
            self._init_route_slot_axis_anchor()
            self._init_lite_polyline_anchor()
            nn.init.zeros_(self.to_route_slot_axis[-1].weight)
            nn.init.zeros_(self.to_route_slot_axis[-1].bias)
            nn.init.zeros_(self.to_route_slot_control[-1].weight)
            nn.init.zeros_(self.to_route_slot_control[-1].bias)
        if topo_proposal_type == 'lane_prior_anchor':
            self._init_lane_prior_axis_anchor()
            nn.init.zeros_(self.to_lane_prior_axis[-1].weight)
            nn.init.zeros_(self.to_lane_prior_axis[-1].bias)
        if topo_proposal_type in ('decomp_endpoint', 'decomp_endpoint_polyline'):
            self._init_decomp_endpoint_polyline_anchor()
            nn.init.zeros_(self.to_endpoint_axis[-1].weight)
            nn.init.zeros_(self.to_endpoint_axis[-1].bias)
        if topo_proposal_type in ('interaction_decomp_endpoint', 'interaction_cv_endpoint'):
            self._init_interaction_axis_anchor()
            nn.init.zeros_(self.to_interaction_axis[-1].weight)
            nn.init.zeros_(self.to_interaction_axis[-1].bias)
        if topo_proposal_type == 'attn_endpoint':
            self._init_attn_endpoint_anchor()
            nn.init.zeros_(self.to_attn_endpoint_delta[-1].weight)
            nn.init.zeros_(self.to_attn_endpoint_delta[-1].bias)
        if topo_proposal_type == 'topo_query':
            self._init_topo_query_endpoint_anchor()
            nn.init.zeros_(self.to_topo_query_endpoint_delta[-1].weight)
            nn.init.zeros_(self.to_topo_query_endpoint_delta[-1].bias)
        if topo_proposal_type == 'decomp_endpoint_polyline':
            nn.init.zeros_(self.to_polyline_control[-1].weight)
            nn.init.zeros_(self.to_polyline_control[-1].bias)
        if topo_proposal_type == 'corridor_mode_endpoint':
            nn.init.zeros_(self.to_corridor_mode_endpoint_delta[-1].weight)
            nn.init.zeros_(self.to_corridor_mode_endpoint_delta[-1].bias)
        if topo_proposal_type in ('corridor_residual', 'corridor_query', 'corridor_query_safe'):
            nn.init.zeros_(self.to_corridor_goal_delta[-1].weight)
            nn.init.zeros_(self.to_corridor_goal_delta[-1].bias)
        if topo_proposal_type in ('corridor_query', 'corridor_query_safe'):
            nn.init.zeros_(self.to_corridor_anchor_delta[-1].weight)
            nn.init.zeros_(self.to_corridor_anchor_delta[-1].bias)

    def forward(self,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor],
                proposal_override: Optional[Mapping[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        agent_state = scene_enc['x_a'][:, -1]
        mode_state = agent_state.unsqueeze(1) + self.mode_emb.weight.unsqueeze(0)
        mode_state = self.query_mlp(mode_state)

        if self.topo_proposal_type == 'corridor_query':
            goal_local, anchor_local = self._make_corridor_query_proposals(data, scene_enc, mode_state)
        elif self.topo_proposal_type == 'corridor_query_safe':
            goal_local, anchor_local = self._make_corridor_query_safe_proposals(data, scene_enc, mode_state)
        elif self.topo_proposal_type == 'corridor_goal':
            goal_local = self._make_corridor_goals(data, scene_enc, mode_state)
            anchor_base = self._make_goal_anchors(goal_local)
            anchor_residual = self.to_anchor_residual(mode_state).view(
                -1, self.num_modes, self.num_future_steps, self.output_dim)
            anchor_local = anchor_base + anchor_residual
        elif self.topo_proposal_type == 'corridor_residual':
            goal_local = self._make_corridor_residual_goals(data, scene_enc, mode_state)
            anchor_base = self._make_goal_anchors(goal_local)
            anchor_residual = self.to_anchor_residual(mode_state).view(
                -1, self.num_modes, self.num_future_steps, self.output_dim)
            anchor_local = anchor_base + anchor_residual
        elif self.topo_proposal_type == 'corridor_mode_endpoint':
            goal_local, anchor_local = self._make_corridor_mode_endpoint_proposals(
                data, scene_enc, agent_state, mode_state)
        elif self.topo_proposal_type == 'decomp_endpoint':
            goal_local = self._make_decomp_endpoint_goals(agent_state, mode_state)
            anchor_base = self._make_goal_anchors(goal_local)
            anchor_residual = self.to_anchor_residual(mode_state).view(
                -1, self.num_modes, self.num_future_steps, self.output_dim)
            anchor_local = anchor_base + anchor_residual
        elif self.topo_proposal_type == 'interaction_decomp_endpoint':
            goal_local = self._make_interaction_decomp_endpoint_goals(data, scene_enc, agent_state, mode_state)
            anchor_base = self._make_goal_anchors(goal_local)
            anchor_residual = self.to_anchor_residual(mode_state).view(
                -1, self.num_modes, self.num_future_steps, self.output_dim)
            anchor_local = anchor_base + anchor_residual
        elif self.topo_proposal_type == 'interaction_cv_endpoint':
            goal_local, anchor_local = self._make_interaction_cv_endpoint_proposals(
                data, scene_enc, agent_state, mode_state)
        elif self.topo_proposal_type == 'attn_endpoint':
            goal_local, anchor_local = self._make_attn_endpoint_proposals(
                data, scene_enc, agent_state, mode_state)
        elif self.topo_proposal_type == 'topo_query':
            goal_local, anchor_local = self._make_topo_query_proposals(
                data, scene_enc, agent_state, mode_state)
        elif self.topo_proposal_type == 'decomp_endpoint_polyline':
            goal_local, anchor_local = self._make_decomp_endpoint_polyline_proposals(agent_state, mode_state)
        elif self.topo_proposal_type == 'mode_endpoint':
            goal_local = self._make_mode_endpoint_goals(agent_state, mode_state)
            anchor_base = self._make_goal_anchors(goal_local)
            anchor_residual = self.to_anchor_residual(mode_state).view(
                -1, self.num_modes, self.num_future_steps, self.output_dim)
            anchor_local = anchor_base + anchor_residual
        elif self.topo_proposal_type == 'mode_endpoint_anchorbasis':
            goal_local = self._make_mode_endpoint_goals(agent_state, mode_state)
            anchor_base = self._make_mode_endpoint_basis_anchors(goal_local)
            anchor_residual = self.to_anchor_residual(mode_state).view(
                -1, self.num_modes, self.num_future_steps, self.output_dim)
            anchor_local = anchor_base + anchor_residual
        elif self.topo_proposal_type == 'mode_endpoint_polyline_readout':
            goal_local, anchor_local = self._make_mode_endpoint_polyline_readout_proposals(
                data, scene_enc, agent_state, mode_state)
        elif self.topo_proposal_type == 'mode_endpoint_polyline_lite':
            goal_local, anchor_local = self._make_mode_endpoint_polyline_lite_proposals(agent_state, mode_state)
        elif self.topo_proposal_type == 'corridor_multi_anchor':
            goal_local, anchor_local = self._make_corridor_multi_anchor_proposals(
                data, scene_enc, agent_state, mode_state)
        elif self.topo_proposal_type == 'route_slot_polyline':
            goal_local, anchor_local = self._make_route_slot_polyline_proposals(
                data, scene_enc, agent_state, mode_state)
        elif self.topo_proposal_type == 'soft_route_slot_polyline':
            goal_local, anchor_local = self._make_soft_route_slot_polyline_proposals(
                data, scene_enc, agent_state, mode_state)
        elif self.topo_proposal_type == 'lane_prior_anchor':
            goal_local, anchor_local = self._make_lane_prior_anchor_proposals(
                data, scene_enc, agent_state, mode_state)
        else:
            goal_local = self.to_goal(mode_state)
            anchor_base = self._make_goal_anchors(goal_local)
            anchor_residual = self.to_anchor_residual(mode_state).view(
                -1, self.num_modes, self.num_future_steps, self.output_dim)
            anchor_local = anchor_base + anchor_residual
        if proposal_override is not None:
            override_anchor = proposal_override['loc_propose_pos'][..., :self.output_dim].detach().to(
                device=anchor_local.device,
                dtype=anchor_local.dtype,
            )
            anchor_local = override_anchor + 0.0 * anchor_local

        corridor_feat, corridor_dist, route_jump = self._extract_corridors(data, scene_enc, anchor_local)
        motion = anchor_local.new_zeros(anchor_local.shape)
        motion[:, :, 1:] = anchor_local[:, :, 1:] - anchor_local[:, :, :-1]
        query_tokens = mode_state.unsqueeze(2).expand(-1, -1, self.num_future_steps, -1)
        agent_tokens = agent_state.unsqueeze(1).unsqueeze(2).expand(-1, self.num_modes, self.num_future_steps, -1)
        corridor_tokens = torch.cat([
            query_tokens,
            agent_tokens,
            corridor_feat,
            anchor_local,
            motion,
            corridor_dist.unsqueeze(-1),
            route_jump.unsqueeze(-1),
        ], dim=-1)

        n, k, t, _ = corridor_tokens.shape
        spatial = self.corridor_token_proj(corridor_tokens).reshape(n * k, t, self.hidden_dim)
        for block in self.spatial_ssm:
            spatial = block(spatial)
        loc_propose_pos = anchor_local + self.to_loc_propose_pos(spatial).reshape(n, k, t, self.output_dim)
        scale_propose_pos = F.elu_(
            self.to_scale_propose_pos(spatial).reshape(n, k, t, self.output_dim),
            alpha=1.0) + 1.0 + 0.1

        propose_motion = loc_propose_pos.new_zeros(loc_propose_pos.shape)
        propose_motion[:, :, 1:] = loc_propose_pos[:, :, 1:] - loc_propose_pos[:, :, :-1]
        rollout_tokens = torch.cat([
            spatial.reshape(n, k, t, self.hidden_dim),
            loc_propose_pos.detach(),
            propose_motion.detach(),
            corridor_dist.unsqueeze(-1),
        ], dim=-1)
        temporal = self.rollout_token_proj(rollout_tokens).reshape(n * k, t, self.hidden_dim)
        for block in self.temporal_ssm:
            temporal = block(temporal)
        loc_refine_pos = loc_propose_pos.detach() + self.to_loc_refine_pos(temporal).reshape(
            n, k, t, self.output_dim)
        scale_refine_pos = F.elu_(
            self.to_scale_refine_pos(temporal).reshape(n, k, t, self.output_dim),
            alpha=1.0) + 1.0 + 0.1

        score_state = temporal.reshape(n, k, t, self.hidden_dim).mean(dim=2)
        score_topology = torch.stack([corridor_dist.mean(dim=-1), route_jump.mean(dim=-1)], dim=-1)
        pi = self.to_pi(torch.cat([score_state, score_topology], dim=-1)).squeeze(-1)
        topo_aux_pi = None
        if self.topo_aux_score:
            aux_state = score_state.detach() if self.topo_aux_score_detach else score_state
            aux_corridor_dist = corridor_dist.detach() if self.topo_aux_score_detach else corridor_dist
            aux_route_jump = route_jump.detach() if self.topo_aux_score_detach else route_jump
            aux_topology = torch.stack([
                aux_corridor_dist.mean(dim=-1),
                aux_corridor_dist.amin(dim=-1),
                aux_route_jump.mean(dim=-1),
                aux_route_jump.amax(dim=-1),
            ], dim=-1)
            topo_aux_pi = self.to_topo_aux_pi(torch.cat([aux_state, aux_topology], dim=-1)).squeeze(-1)

        loc_propose_head, conc_propose_head = self._heads_from_positions(loc_propose_pos, scale_propose_pos)
        loc_refine_head, conc_refine_head = self._heads_from_positions(loc_refine_pos, scale_refine_pos)

        out = {
            'loc_propose_pos': loc_propose_pos,
            'scale_propose_pos': scale_propose_pos,
            'loc_propose_head': loc_propose_head,
            'conc_propose_head': conc_propose_head,
            'loc_refine_pos': loc_refine_pos,
            'scale_refine_pos': scale_refine_pos,
            'loc_refine_head': loc_refine_head,
            'conc_refine_head': conc_refine_head,
            'pi': pi,
            'topo_corridor_dist': corridor_dist,
            'topo_route_jump': route_jump,
        }
        if topo_aux_pi is not None:
            out['topo_aux_pi'] = topo_aux_pi
        return out

    def _make_goal_anchors(self, goal_local: torch.Tensor) -> torch.Tensor:
        alpha = torch.linspace(
            1.0 / self.num_future_steps,
            1.0,
            self.num_future_steps,
            device=goal_local.device,
            dtype=goal_local.dtype,
        )
        return goal_local.unsqueeze(2) * alpha.view(1, 1, self.num_future_steps, 1)

    def _make_mode_endpoint_basis_anchors(self, goal_local: torch.Tensor) -> torch.Tensor:
        goal_dir, goal_ortho = self._build_goal_basis(goal_local)
        goal_norm = goal_local.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        alpha = torch.linspace(
            1.0 / self.num_future_steps,
            1.0,
            self.num_future_steps,
            device=goal_local.device,
            dtype=goal_local.dtype,
        ).view(1, 1, self.num_future_steps, 1)
        coeff = goal_local.new_zeros((1, self.num_modes, self.num_future_steps, 2))
        coeff[..., 0:1] = alpha
        basis_offset = self.topo_anchor_basis_scale * torch.tanh(
            self.mode_anchor_basis_offset.to(device=goal_local.device, dtype=goal_local.dtype)
        ).unsqueeze(0)
        basis_offset = basis_offset.clone()
        basis_offset[..., -1, :] = 0.0
        coeff = coeff + basis_offset
        return goal_norm.unsqueeze(2) * (
            coeff[..., 0:1] * goal_dir.unsqueeze(2) +
            coeff[..., 1:2] * goal_ortho.unsqueeze(2)
        )

    def _init_mode_endpoint_anchor(self) -> None:
        with torch.no_grad():
            progress = torch.linspace(-1.0, 1.0, self.num_modes, dtype=self.mode_endpoint_anchor.dtype)
            self.mode_endpoint_anchor.zero_()
            self.mode_endpoint_anchor[:, 0] = 0.5 * progress
            if self.output_dim > 1:
                self.mode_endpoint_anchor[:, 1] = progress

    def _init_readout_polyline_anchor(self) -> None:
        with torch.no_grad():
            progress = torch.linspace(-1.0, 1.0, self.num_modes, dtype=self.polyline_control_anchor.dtype)
            self.polyline_control_anchor.zero_()
            self.polyline_control_anchor[:, 0, 1] = 0.35 * progress
            self.polyline_control_anchor[:, 1, 1] = 0.70 * progress

    def _init_lite_polyline_anchor(self) -> None:
        with torch.no_grad():
            progress = torch.linspace(-1.0, 1.0, self.num_modes, dtype=self.polyline_control_anchor.dtype)
            self.polyline_control_anchor.zero_()
            self.polyline_control_anchor[:, 0, 1] = 0.45 * progress

    def _init_route_slot_axis_anchor(self) -> None:
        with torch.no_grad():
            progress = torch.linspace(-1.0, 1.0, self.num_modes, dtype=self.route_slot_axis_anchor.dtype)
            self.route_slot_axis_anchor.zero_()
            self.route_slot_axis_anchor[:, 0] = 0.15 * progress
            self.route_slot_axis_anchor[:, 1] = 0.60 * progress

    def _init_lane_prior_axis_anchor(self) -> None:
        with torch.no_grad():
            progress = torch.linspace(-1.0, 1.0, self.num_modes, dtype=self.lane_prior_axis_anchor.dtype)
            self.lane_prior_axis_anchor.zero_()
            self.lane_prior_axis_anchor[:, 0] = 0.10 * progress
            self.lane_prior_axis_anchor[:, 1] = 0.50 * progress

    def _init_decomp_endpoint_polyline_anchor(self) -> None:
        with torch.no_grad():
            progress = torch.linspace(-1.0, 1.0, self.num_modes, dtype=self.endpoint_axis_anchor.dtype)
            self.endpoint_axis_anchor.zero_()
            self.endpoint_axis_anchor[:, 0] = 0.25 * progress
            self.endpoint_axis_anchor[:, 1] = progress
            if hasattr(self, 'polyline_control_anchor'):
                self.polyline_control_anchor.zero_()
                self.polyline_control_anchor[:, 0, 1] = 0.5 * progress
                self.polyline_control_anchor[:, 1, 1] = progress

    def _init_interaction_axis_anchor(self) -> None:
        with torch.no_grad():
            progress = torch.linspace(-1.0, 1.0, self.num_modes, dtype=self.interaction_axis_anchor.dtype)
            self.interaction_axis_anchor.zero_()
            self.interaction_axis_anchor[:, 0] = 0.20 * progress
            self.interaction_axis_anchor[:, 1] = 0.80 * progress

    def _init_attn_endpoint_anchor(self) -> None:
        with torch.no_grad():
            progress = torch.linspace(-1.0, 1.0, self.num_modes, dtype=self.attn_endpoint_anchor.dtype)
            self.attn_endpoint_anchor.zero_()
            self.attn_endpoint_anchor[:, 0] = 0.30 * progress
            self.attn_endpoint_anchor[:, 1] = progress

    def _init_topo_query_endpoint_anchor(self) -> None:
        with torch.no_grad():
            progress = torch.linspace(-1.0, 1.0, self.num_modes, dtype=self.topo_query_endpoint_anchor.dtype)
            self.topo_query_endpoint_anchor.zero_()
            self.topo_query_endpoint_anchor[:, 0] = 0.35 * progress
            self.topo_query_endpoint_anchor[:, 1] = progress

    def _make_mode_endpoint_goals(self,
                                  agent_state: torch.Tensor,
                                  mode_state: torch.Tensor) -> torch.Tensor:
        fallback_goal = self.to_goal(mode_state)
        agent_tokens = agent_state.unsqueeze(1).expand(-1, self.num_modes, -1)
        mode_anchor = self.mode_endpoint_anchor.unsqueeze(0).expand(mode_state.size(0), -1, -1)
        delta_input = torch.cat([mode_state, agent_tokens, fallback_goal, mode_anchor], dim=-1)
        endpoint_delta = mode_anchor + self.to_mode_endpoint_delta(delta_input)
        endpoint_delta = self.topo_mode_endpoint_scale * torch.tanh(endpoint_delta)
        return fallback_goal + endpoint_delta

    def _make_mode_endpoint_polyline_readout_proposals(self,
                                                       data: HeteroData,
                                                       scene_enc: Mapping[str, torch.Tensor],
                                                       agent_state: torch.Tensor,
                                                       mode_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_goal = self._make_mode_endpoint_goals(agent_state, mode_state)
        anchor_residual = self.to_anchor_residual(mode_state).view(
            -1, self.num_modes, self.num_future_steps, self.output_dim)
        rough_anchor = self._make_goal_anchors(base_goal) + anchor_residual

        if data['map_polygon']['position'].numel() == 0:
            return base_goal, rough_anchor

        corridor_feat, corridor_dist, route_jump = self._extract_corridors(data, scene_enc, rough_anchor)
        agent_tokens = agent_state.unsqueeze(1).expand(-1, self.num_modes, -1)
        corridor_summary = corridor_feat.mean(dim=2)
        endpoint_feat = corridor_feat[:, :, -1]
        endpoint_dist = corridor_dist[:, :, -1:].to(dtype=mode_state.dtype)
        endpoint_jump = route_jump[:, :, -1:].to(dtype=mode_state.dtype)
        mode_anchor = self.mode_endpoint_anchor.unsqueeze(0).expand(mode_state.size(0), -1, -1)

        goal_delta_input = torch.cat([
            mode_state,
            agent_tokens,
            corridor_summary,
            endpoint_feat,
            base_goal,
            mode_anchor,
            endpoint_dist,
            endpoint_jump,
        ], dim=-1)
        goal_delta = self.topo_goal_residual_scale * torch.tanh(self.to_readout_goal_delta(goal_delta_input))
        blend = max(0.0, min(float(self.topo_goal_anchor_blend), 1.0))
        goal_local = base_goal + blend * goal_delta

        goal_dir, goal_ortho = self._build_goal_basis(goal_local)
        control_anchor = self.polyline_control_anchor.unsqueeze(0).expand(mode_state.size(0), -1, -1, -1)
        control_input = torch.cat([
            mode_state,
            agent_tokens,
            corridor_summary,
            endpoint_feat,
            goal_local,
            goal_delta,
            endpoint_dist,
            endpoint_jump,
            control_anchor.reshape(mode_state.size(0), self.num_modes, -1),
        ], dim=-1)
        control_axis = control_anchor + self.to_readout_polyline_control(control_input).view(
            mode_state.size(0), self.num_modes, self.num_polyline_control_points, 2)
        control_axis = self.topo_polyline_control_scale * torch.tanh(control_axis)
        control_delta = (
            control_axis[..., :1] * goal_dir.unsqueeze(2) +
            control_axis[..., 1:2] * goal_ortho.unsqueeze(2)
        )
        control_alpha = torch.linspace(
            1.0 / (self.num_polyline_control_points + 1),
            self.num_polyline_control_points / (self.num_polyline_control_points + 1),
            self.num_polyline_control_points,
            device=goal_local.device,
            dtype=goal_local.dtype,
        ).view(1, 1, -1, 1)
        control_local = control_alpha * goal_local.unsqueeze(2) + control_delta
        anchor_base = self._make_piecewise_polyline_anchors(goal_local, control_local)
        anchor_local = anchor_base + anchor_residual
        return goal_local, anchor_local

    def _make_mode_endpoint_polyline_lite_proposals(self,
                                                    agent_state: torch.Tensor,
                                                    mode_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        goal_local = self._make_mode_endpoint_goals(agent_state, mode_state)
        anchor_residual = self.to_anchor_residual(mode_state).view(
            -1, self.num_modes, self.num_future_steps, self.output_dim)
        agent_tokens = agent_state.unsqueeze(1).expand(-1, self.num_modes, -1)
        control_anchor = self.polyline_control_anchor.unsqueeze(0).expand(mode_state.size(0), -1, -1, -1)
        control_input = torch.cat([
            mode_state,
            agent_tokens,
            goal_local,
            control_anchor.reshape(mode_state.size(0), self.num_modes, -1),
        ], dim=-1)
        control_axis = control_anchor + self.to_polyline_control_lite(control_input).view(
            mode_state.size(0), self.num_modes, self.num_polyline_control_points, 2)
        control_axis = self.topo_polyline_control_scale * torch.tanh(control_axis)
        goal_dir, goal_ortho = self._build_goal_basis(goal_local)
        control_delta = (
            control_axis[..., :1] * goal_dir.unsqueeze(2) +
            control_axis[..., 1:2] * goal_ortho.unsqueeze(2)
        )
        control_alpha = goal_local.new_full((1, 1, 1, 1), 0.5)
        control_local = control_alpha * goal_local.unsqueeze(2) + control_delta
        anchor_base = self._make_piecewise_polyline_anchors(goal_local, control_local)
        anchor_local = anchor_base + anchor_residual
        return goal_local, anchor_local

    def _make_corridor_multi_anchor_proposals(self,
                                              data: HeteroData,
                                              scene_enc: Mapping[str, torch.Tensor],
                                              agent_state: torch.Tensor,
                                              mode_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = mode_state.device
        dtype = mode_state.dtype
        base_goal = self._make_mode_endpoint_goals(agent_state, mode_state)
        fallback_anchor = self._make_goal_anchors(base_goal) + self.to_anchor_residual(mode_state).view(
            -1, self.num_modes, self.num_future_steps, self.output_dim)

        map_pos = data['map_polygon']['position'][:, :2].to(device=device, dtype=dtype)
        if map_pos.numel() == 0:
            return base_goal, fallback_anchor

        map_feat = scene_enc['x_pl'][:, -1].to(device=device, dtype=dtype)
        agent_origin = data['agent']['position'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        agent_heading = data['agent']['heading'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        goal_local = base_goal.new_empty(base_goal.shape)
        anchor_local = fallback_anchor.new_empty(fallback_anchor.shape)

        if isinstance(data, Batch):
            agent_batch = data['agent']['batch'].to(device=device)
            map_batch = data['map_polygon']['batch'].to(device=device)
            for batch_id in torch.unique(agent_batch, sorted=True):
                agent_idx = torch.nonzero(agent_batch == batch_id, as_tuple=False).flatten()
                map_idx = torch.nonzero(map_batch == batch_id, as_tuple=False).flatten()
                if agent_idx.numel() == 0:
                    continue
                if map_idx.numel() == 0:
                    goal_local[agent_idx] = base_goal[agent_idx]
                    anchor_local[agent_idx] = fallback_anchor[agent_idx]
                    continue
                goal_part, anchor_part = self._select_corridor_multi_anchor_for_agents(
                    mode_state[agent_idx],
                    base_goal[agent_idx],
                    fallback_anchor[agent_idx],
                    agent_origin[agent_idx],
                    agent_heading[agent_idx],
                    map_pos[map_idx],
                    map_feat[map_idx],
                )
                goal_local[agent_idx] = goal_part
                anchor_local[agent_idx] = anchor_part
        else:
            goal_local, anchor_local = self._select_corridor_multi_anchor_for_agents(
                mode_state,
                base_goal,
                fallback_anchor,
                agent_origin,
                agent_heading,
                map_pos,
                map_feat,
            )
        return goal_local, anchor_local

    def _select_corridor_multi_anchor_for_agents(self,
                                                 mode_state: torch.Tensor,
                                                 base_goal: torch.Tensor,
                                                 fallback_anchor: torch.Tensor,
                                                 agent_origin: torch.Tensor,
                                                 agent_heading: torch.Tensor,
                                                 map_pos: torch.Tensor,
                                                 map_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_global = self._local_to_global(
            base_goal.unsqueeze(2),
            agent_origin[:, :2],
            agent_heading,
        ).squeeze(2)
        distance = torch.cdist(base_global.reshape(-1, self.output_dim), map_pos).reshape(
            mode_state.size(0), self.num_modes, -1)
        topk = min(max(self.num_modes, 8), distance.size(-1))
        nearest_dist, nearest_idx = distance.topk(k=topk, dim=-1, largest=False)
        query_score_all = torch.einsum('nkh,mh->nkm', mode_state, map_feat) / math.sqrt(self.hidden_dim)
        query_score = query_score_all.gather(dim=-1, index=nearest_idx)
        local_score = query_score - self.topo_goal_distance_weight * nearest_dist / self.corridor_dist_norm
        corridor_order = local_score.topk(k=topk, dim=-1, largest=True).indices

        mode_rank = torch.linspace(
            0,
            topk - 1,
            self.num_modes,
            device=mode_state.device,
        ).round().long().view(1, self.num_modes, 1)
        endpoint_rank = corridor_order.gather(
            dim=-1,
            index=mode_rank.expand(mode_state.size(0), -1, -1),
        )
        support_rank = corridor_order.gather(
            dim=-1,
            index=(mode_rank + 1).clamp(max=topk - 1).expand(mode_state.size(0), -1, -1),
        )

        endpoint_idx = nearest_idx.gather(dim=-1, index=endpoint_rank).squeeze(-1)
        support_idx = nearest_idx.gather(dim=-1, index=support_rank).squeeze(-1)
        endpoint_local = self._global_to_local(map_pos[endpoint_idx], agent_origin[:, :2], agent_heading)
        support_local = self._global_to_local(map_pos[support_idx], agent_origin[:, :2], agent_heading)
        endpoint_feat = map_feat[endpoint_idx]
        support_feat = map_feat[support_idx]

        endpoint_offset = endpoint_local - base_goal
        goal_delta_input = torch.cat([mode_state, endpoint_feat, support_feat, base_goal, endpoint_offset], dim=-1)
        goal_delta = self.topo_goal_residual_scale * torch.tanh(self.to_corridor_multi_goal_delta(goal_delta_input))
        blend = max(0.0, min(float(self.topo_goal_anchor_blend), 1.0))
        goal_local = base_goal + blend * ((endpoint_local + goal_delta) - base_goal)

        goal_dir, goal_ortho = self._build_goal_basis(goal_local)
        control_anchor = self.polyline_control_anchor.unsqueeze(0).expand(mode_state.size(0), -1, -1, -1)
        support_center = 0.5 * (support_local + 0.5 * goal_local)
        control_input = torch.cat([
            mode_state,
            endpoint_feat,
            support_feat,
            goal_local,
            endpoint_local,
            support_local,
            control_anchor.reshape(mode_state.size(0), self.num_modes, -1),
        ], dim=-1)
        control_axis = control_anchor + self.to_corridor_multi_control(control_input).view(
            mode_state.size(0), self.num_modes, self.num_polyline_control_points, 2)
        control_axis = self.topo_polyline_control_scale * torch.tanh(control_axis)
        control_delta = (
            control_axis[..., :1] * goal_dir.unsqueeze(2) +
            control_axis[..., 1:2] * goal_ortho.unsqueeze(2)
        )
        control_local = support_center.unsqueeze(2) + control_delta

        anchor_base = self._make_piecewise_polyline_anchors(goal_local, control_local)
        anchor_residual = self.to_anchor_residual(mode_state).view(
            -1, self.num_modes, self.num_future_steps, self.output_dim)
        anchor_local = anchor_base + anchor_residual
        return goal_local, anchor_local

    def _make_route_slot_polyline_proposals(self,
                                            data: HeteroData,
                                            scene_enc: Mapping[str, torch.Tensor],
                                            agent_state: torch.Tensor,
                                            mode_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_goal = self._make_mode_endpoint_goals(agent_state, mode_state)
        fallback_anchor = self._make_goal_anchors(base_goal) + self.to_anchor_residual(mode_state).view(
            -1, self.num_modes, self.num_future_steps, self.output_dim)
        device = mode_state.device
        dtype = mode_state.dtype
        map_pos = data['map_polygon']['position'][:, :2].to(device=device, dtype=dtype)
        if map_pos.numel() == 0:
            return base_goal, fallback_anchor

        map_feat = scene_enc['x_pl'][:, -1].to(device=device, dtype=dtype)
        agent_origin = data['agent']['position'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        agent_heading = data['agent']['heading'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        goal_local = base_goal.new_empty(base_goal.shape)
        anchor_local = fallback_anchor.new_empty(fallback_anchor.shape)

        if isinstance(data, Batch):
            agent_batch = data['agent']['batch'].to(device=device)
            map_batch = data['map_polygon']['batch'].to(device=device)
            for batch_id in torch.unique(agent_batch, sorted=True):
                agent_idx = torch.nonzero(agent_batch == batch_id, as_tuple=False).flatten()
                map_idx = torch.nonzero(map_batch == batch_id, as_tuple=False).flatten()
                if agent_idx.numel() == 0:
                    continue
                if map_idx.numel() == 0:
                    goal_local[agent_idx] = base_goal[agent_idx]
                    anchor_local[agent_idx] = fallback_anchor[agent_idx]
                    continue
                goal_part, anchor_part = self._select_route_slot_for_agents(
                    mode_state[agent_idx],
                    agent_state[agent_idx],
                    base_goal[agent_idx],
                    fallback_anchor[agent_idx],
                    agent_origin[agent_idx],
                    agent_heading[agent_idx],
                    map_pos[map_idx],
                    map_feat[map_idx],
                )
                goal_local[agent_idx] = goal_part
                anchor_local[agent_idx] = anchor_part
        else:
            goal_local, anchor_local = self._select_route_slot_for_agents(
                mode_state,
                agent_state,
                base_goal,
                fallback_anchor,
                agent_origin,
                agent_heading,
                map_pos,
                map_feat,
            )
        return goal_local, anchor_local

    def _make_soft_route_slot_polyline_proposals(self,
                                                 data: HeteroData,
                                                 scene_enc: Mapping[str, torch.Tensor],
                                                 agent_state: torch.Tensor,
                                                 mode_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_goal = self._make_mode_endpoint_goals(agent_state, mode_state)
        fallback_anchor = self._make_goal_anchors(base_goal) + self.to_anchor_residual(mode_state).view(
            -1, self.num_modes, self.num_future_steps, self.output_dim)
        device = mode_state.device
        dtype = mode_state.dtype
        map_pos = data['map_polygon']['position'][:, :2].to(device=device, dtype=dtype)
        if map_pos.numel() == 0:
            return base_goal, fallback_anchor

        map_feat = scene_enc['x_pl'][:, -1].to(device=device, dtype=dtype)
        agent_origin = data['agent']['position'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        agent_heading = data['agent']['heading'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        goal_local = base_goal.new_empty(base_goal.shape)
        anchor_local = fallback_anchor.new_empty(fallback_anchor.shape)

        if isinstance(data, Batch):
            agent_batch = data['agent']['batch'].to(device=device)
            map_batch = data['map_polygon']['batch'].to(device=device)
            for batch_id in torch.unique(agent_batch, sorted=True):
                agent_idx = torch.nonzero(agent_batch == batch_id, as_tuple=False).flatten()
                map_idx = torch.nonzero(map_batch == batch_id, as_tuple=False).flatten()
                if agent_idx.numel() == 0:
                    continue
                if map_idx.numel() == 0:
                    goal_local[agent_idx] = base_goal[agent_idx]
                    anchor_local[agent_idx] = fallback_anchor[agent_idx]
                    continue
                goal_part, anchor_part = self._select_soft_route_slot_for_agents(
                    mode_state[agent_idx],
                    agent_state[agent_idx],
                    base_goal[agent_idx],
                    fallback_anchor[agent_idx],
                    agent_origin[agent_idx],
                    agent_heading[agent_idx],
                    map_pos[map_idx],
                    map_feat[map_idx],
                )
                goal_local[agent_idx] = goal_part
                anchor_local[agent_idx] = anchor_part
        else:
            goal_local, anchor_local = self._select_soft_route_slot_for_agents(
                mode_state,
                agent_state,
                base_goal,
                fallback_anchor,
                agent_origin,
                agent_heading,
                map_pos,
                map_feat,
            )
        return goal_local, anchor_local

    def _select_route_slot_for_agents(self,
                                      mode_state: torch.Tensor,
                                      agent_state: torch.Tensor,
                                      base_goal: torch.Tensor,
                                      fallback_anchor: torch.Tensor,
                                      agent_origin: torch.Tensor,
                                      agent_heading: torch.Tensor,
                                      map_pos: torch.Tensor,
                                      map_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_agents = mode_state.size(0)
        base_global = self._local_to_global(base_goal.unsqueeze(2), agent_origin[:, :2], agent_heading).squeeze(2)
        distance = torch.cdist(base_global.reshape(-1, self.output_dim), map_pos).reshape(
            num_agents, self.num_modes, -1)
        topk = min(max(self.num_modes * 2, int(self.topo_route_slot_topk)), distance.size(-1))
        nearest_dist, nearest_idx = distance.topk(k=topk, dim=-1, largest=False)

        query_score_all = torch.einsum('nkh,mh->nkm', mode_state, map_feat) / math.sqrt(self.hidden_dim)
        query_score = query_score_all.gather(dim=-1, index=nearest_idx)

        candidate_global = map_pos[nearest_idx.reshape(-1)].view(num_agents, self.num_modes, topk, self.output_dim)
        candidate_local = self._global_to_local(
            candidate_global.reshape(-1, self.output_dim),
            agent_origin[:, None, None, :2].expand(-1, self.num_modes, topk, -1).reshape(-1, self.output_dim),
            agent_heading[:, None, None].expand(-1, self.num_modes, topk).reshape(-1),
        ).view(num_agents, self.num_modes, topk, self.output_dim)
        forward_score = candidate_local[..., 0] / self.corridor_dist_norm
        lateral_penalty = candidate_local[..., 1].abs() / self.corridor_dist_norm
        local_score = (
            query_score
            - self.topo_goal_distance_weight * nearest_dist / self.corridor_dist_norm
            + 0.05 * forward_score
            - 0.02 * lateral_penalty
        )
        best_rank = local_score.argmax(dim=-1, keepdim=True)
        slot_idx = nearest_idx.gather(dim=-1, index=best_rank).squeeze(-1)
        slot_local = candidate_local.gather(
            dim=2,
            index=best_rank.unsqueeze(-1).expand(-1, -1, 1, self.output_dim),
        ).squeeze(2)
        slot_feat = map_feat[slot_idx]

        agent_tokens = agent_state.unsqueeze(1).expand(-1, self.num_modes, -1)
        axis_anchor = self.route_slot_axis_anchor.unsqueeze(0).expand(num_agents, -1, -1)
        axis_input = torch.cat([mode_state, agent_tokens, slot_feat, slot_local, base_goal, axis_anchor], dim=-1)
        axis_coeff = axis_anchor + self.to_route_slot_axis(axis_input)
        axis_coeff = torch.tanh(axis_coeff)
        slot_dir, slot_ortho = self._build_goal_basis(slot_local)
        slot_blend = max(0.0, min(float(self.topo_goal_anchor_blend), 1.0))
        slot_center = slot_blend * slot_local + (1.0 - slot_blend) * base_goal
        axis_delta = (
            self.topo_route_slot_longitudinal_scale * axis_coeff[..., :1] * slot_dir +
            self.topo_route_slot_lateral_scale * axis_coeff[..., 1:2] * slot_ortho
        )
        goal_local = slot_center + axis_delta

        control_anchor = self.polyline_control_anchor.unsqueeze(0).expand(num_agents, -1, -1, -1)
        control_input = torch.cat([
            mode_state,
            agent_tokens,
            slot_feat,
            goal_local,
            slot_local,
            axis_coeff,
            control_anchor.reshape(num_agents, self.num_modes, -1),
        ], dim=-1)
        control_axis = control_anchor + self.to_route_slot_control(control_input).view(
            num_agents, self.num_modes, self.num_polyline_control_points, 2)
        control_axis = self.topo_polyline_control_scale * torch.tanh(control_axis)
        control_delta = (
            control_axis[..., :1] * slot_dir.unsqueeze(2) +
            control_axis[..., 1:2] * slot_ortho.unsqueeze(2)
        )
        control_center = 0.5 * slot_center
        control_local = control_center.unsqueeze(2) + control_delta

        anchor_base = self._make_piecewise_polyline_anchors(goal_local, control_local)
        anchor_residual = self.to_anchor_residual(mode_state).view(
            -1, self.num_modes, self.num_future_steps, self.output_dim)
        anchor_local = anchor_base + anchor_residual
        return goal_local, anchor_local

    def _select_soft_route_slot_for_agents(self,
                                           mode_state: torch.Tensor,
                                           agent_state: torch.Tensor,
                                           base_goal: torch.Tensor,
                                           fallback_anchor: torch.Tensor,
                                           agent_origin: torch.Tensor,
                                           agent_heading: torch.Tensor,
                                           map_pos: torch.Tensor,
                                           map_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_agents = mode_state.size(0)
        base_global = self._local_to_global(base_goal.unsqueeze(2), agent_origin[:, :2], agent_heading).squeeze(2)
        distance = torch.cdist(base_global.reshape(-1, self.output_dim), map_pos).reshape(
            num_agents, self.num_modes, -1)
        topk = min(max(self.num_modes * 2, int(self.topo_route_slot_topk)), distance.size(-1))
        nearest_dist, nearest_idx = distance.topk(k=topk, dim=-1, largest=False)

        query_score_all = torch.einsum('nkh,mh->nkm', mode_state, map_feat) / math.sqrt(self.hidden_dim)
        query_score = query_score_all.gather(dim=-1, index=nearest_idx)
        candidate_global = map_pos[nearest_idx.reshape(-1)].view(num_agents, self.num_modes, topk, self.output_dim)
        candidate_local = self._global_to_local(
            candidate_global.reshape(-1, self.output_dim),
            agent_origin[:, None, None, :2].expand(-1, self.num_modes, topk, -1).reshape(-1, self.output_dim),
            agent_heading[:, None, None].expand(-1, self.num_modes, topk).reshape(-1),
        ).view(num_agents, self.num_modes, topk, self.output_dim)
        candidate_feat = map_feat[nearest_idx.reshape(-1)].view(num_agents, self.num_modes, topk, self.hidden_dim)
        forward_score = candidate_local[..., 0] / self.corridor_dist_norm
        lateral_penalty = candidate_local[..., 1].abs() / self.corridor_dist_norm
        local_score = (
            query_score
            - self.topo_goal_distance_weight * nearest_dist / self.corridor_dist_norm
            + 0.05 * forward_score
            - 0.02 * lateral_penalty
        )
        temperature = max(float(self.topo_route_slot_soft_temperature), 1e-3)
        slot_weight = torch.softmax(local_score / temperature, dim=-1)
        slot_local = (slot_weight.unsqueeze(-1) * candidate_local).sum(dim=2)
        slot_feat = (slot_weight.unsqueeze(-1) * candidate_feat).sum(dim=2)

        agent_tokens = agent_state.unsqueeze(1).expand(-1, self.num_modes, -1)
        axis_anchor = self.route_slot_axis_anchor.unsqueeze(0).expand(num_agents, -1, -1)
        axis_input = torch.cat([mode_state, agent_tokens, slot_feat, slot_local, base_goal, axis_anchor], dim=-1)
        axis_coeff = axis_anchor + self.to_route_slot_axis(axis_input)
        axis_coeff = torch.tanh(axis_coeff)
        slot_dir, slot_ortho = self._build_goal_basis(slot_local)
        slot_blend = max(0.0, min(float(self.topo_goal_anchor_blend), 1.0))
        slot_center = slot_blend * slot_local + (1.0 - slot_blend) * base_goal
        axis_delta = (
            self.topo_route_slot_longitudinal_scale * axis_coeff[..., :1] * slot_dir +
            self.topo_route_slot_lateral_scale * axis_coeff[..., 1:2] * slot_ortho
        )
        goal_local = slot_center + axis_delta

        control_anchor = self.polyline_control_anchor.unsqueeze(0).expand(num_agents, -1, -1, -1)
        control_input = torch.cat([
            mode_state,
            agent_tokens,
            slot_feat,
            goal_local,
            slot_local,
            axis_coeff,
            control_anchor.reshape(num_agents, self.num_modes, -1),
        ], dim=-1)
        control_axis = control_anchor + self.to_route_slot_control(control_input).view(
            num_agents, self.num_modes, self.num_polyline_control_points, 2)
        control_axis = self.topo_polyline_control_scale * torch.tanh(control_axis)
        control_delta = (
            control_axis[..., :1] * slot_dir.unsqueeze(2) +
            control_axis[..., 1:2] * slot_ortho.unsqueeze(2)
        )
        control_center = 0.5 * slot_center
        control_local = control_center.unsqueeze(2) + control_delta

        anchor_base = self._make_piecewise_polyline_anchors(goal_local, control_local)
        anchor_residual = self.to_anchor_residual(mode_state).view(
            -1, self.num_modes, self.num_future_steps, self.output_dim)
        anchor_local = anchor_base + anchor_residual
        return goal_local, anchor_local

    def _make_lane_prior_anchor_proposals(self,
                                          data: HeteroData,
                                          scene_enc: Mapping[str, torch.Tensor],
                                          agent_state: torch.Tensor,
                                          mode_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = mode_state.device
        dtype = mode_state.dtype
        fallback_goal = self.to_goal(mode_state)
        cv_anchor = self._make_cv_anchors(data, device, dtype)
        cv_goal = cv_anchor[:, -1].unsqueeze(1).expand(-1, self.num_modes, -1)

        map_pos = data['map_polygon']['position'][:, :2].to(device=device, dtype=dtype)
        if map_pos.numel() == 0:
            anchor_residual = self.to_anchor_residual(mode_state).view(
                -1, self.num_modes, self.num_future_steps, self.output_dim)
            return cv_goal, cv_anchor.unsqueeze(1) + anchor_residual

        map_feat = scene_enc['x_pl'][:, -1].to(device=device, dtype=dtype)
        agent_origin = data['agent']['position'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        agent_heading = data['agent']['heading'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        goal_local = cv_goal.new_empty(cv_goal.shape)
        anchor_local = cv_goal.new_empty(
            (mode_state.size(0), self.num_modes, self.num_future_steps, self.output_dim))

        if isinstance(data, Batch):
            agent_batch = data['agent']['batch'].to(device=device)
            map_batch = data['map_polygon']['batch'].to(device=device)
            for batch_id in torch.unique(agent_batch, sorted=True):
                agent_idx = torch.nonzero(agent_batch == batch_id, as_tuple=False).flatten()
                map_idx = torch.nonzero(map_batch == batch_id, as_tuple=False).flatten()
                if agent_idx.numel() == 0:
                    continue
                if map_idx.numel() == 0:
                    anchor_residual = self.to_anchor_residual(mode_state[agent_idx]).view(
                        -1, self.num_modes, self.num_future_steps, self.output_dim)
                    goal_local[agent_idx] = cv_goal[agent_idx]
                    anchor_local[agent_idx] = cv_anchor[agent_idx].unsqueeze(1) + anchor_residual
                    continue
                goal_part, anchor_part = self._select_lane_prior_anchor_for_agents(
                    mode_state[agent_idx],
                    agent_state[agent_idx],
                    cv_goal[agent_idx],
                    cv_anchor[agent_idx],
                    agent_origin[agent_idx],
                    agent_heading[agent_idx],
                    map_pos[map_idx],
                    map_feat[map_idx],
                )
                goal_local[agent_idx] = goal_part
                anchor_local[agent_idx] = anchor_part
        else:
            goal_local, anchor_local = self._select_lane_prior_anchor_for_agents(
                mode_state,
                agent_state,
                cv_goal,
                cv_anchor,
                agent_origin,
                agent_heading,
                map_pos,
                map_feat,
            )
        goal_local = goal_local + 0.0 * fallback_goal
        return goal_local, anchor_local

    def _select_lane_prior_anchor_for_agents(self,
                                             mode_state: torch.Tensor,
                                             agent_state: torch.Tensor,
                                             cv_goal: torch.Tensor,
                                             cv_anchor: torch.Tensor,
                                             agent_origin: torch.Tensor,
                                             agent_heading: torch.Tensor,
                                             map_pos: torch.Tensor,
                                             map_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_agents = mode_state.size(0)
        map_local = self._global_to_local(
            map_pos.unsqueeze(0).expand(num_agents, -1, -1),
            agent_origin[:, :2],
            agent_heading,
        )
        cv_goal_single = cv_goal[:, 0]
        forward = map_local[..., 0]
        lateral = map_local[..., 1].abs()
        cv_dist = (map_local - cv_goal_single.unsqueeze(1)).norm(dim=-1)
        target_forward = cv_goal_single[..., 0].clamp_min(5.0).unsqueeze(1)
        reach_penalty = (forward - target_forward).abs() / self.corridor_dist_norm
        forward_bonus = forward.clamp(min=0.0, max=self.corridor_dist_norm) / self.corridor_dist_norm
        geometry_score = (
            -0.70 * reach_penalty
            -0.25 * lateral / self.corridor_dist_norm
            -0.25 * cv_dist / self.corridor_dist_norm
            +0.15 * forward_bonus
        )
        geometry_score = geometry_score.masked_fill(forward < 0.5, -1e4)

        query_score = torch.einsum('nkh,mh->nkm', mode_state, map_feat) / math.sqrt(self.hidden_dim)
        score = geometry_score.unsqueeze(1) + 0.10 * query_score
        topk = min(max(self.num_modes * 4, int(self.topo_route_slot_topk)), map_pos.size(0))
        top_score, top_idx = score.topk(k=topk, dim=-1, largest=True)

        # Spread modes across the strongest reachable lane candidates instead of letting all modes collapse.
        mode_rank = torch.linspace(
            0,
            topk - 1,
            self.num_modes,
            device=mode_state.device,
        ).round().long().view(1, self.num_modes, 1)
        sorted_rank = top_score.argsort(dim=-1, descending=True)
        picked_rank = sorted_rank.gather(
            dim=-1,
            index=mode_rank.expand(num_agents, -1, -1),
        )
        slot_idx = top_idx.gather(dim=-1, index=picked_rank).squeeze(-1)
        slot_local = map_local.gather(
            dim=1,
            index=slot_idx.reshape(num_agents, self.num_modes, 1).expand(-1, -1, self.output_dim),
        )
        slot_feat = map_feat[slot_idx]

        agent_tokens = agent_state.unsqueeze(1).expand(-1, self.num_modes, -1)
        axis_anchor = self.lane_prior_axis_anchor.unsqueeze(0).expand(num_agents, -1, -1)
        axis_input = torch.cat([mode_state, agent_tokens, slot_feat, slot_local, cv_goal, axis_anchor], dim=-1)
        axis_coeff = torch.tanh(axis_anchor + self.to_lane_prior_axis(axis_input))
        slot_dir, slot_ortho = self._build_goal_basis(slot_local)
        axis_delta = (
            self.topo_route_slot_longitudinal_scale * axis_coeff[..., :1] * slot_dir
            + self.topo_route_slot_lateral_scale * axis_coeff[..., 1:2] * slot_ortho
        )
        goal_local = slot_local + axis_delta

        anchor_base = cv_anchor.unsqueeze(1) + self._make_goal_anchors(goal_local - cv_goal)
        anchor_residual = self.to_anchor_residual(mode_state).view(
            -1, self.num_modes, self.num_future_steps, self.output_dim)
        anchor_local = anchor_base + 0.25 * anchor_residual
        return goal_local, anchor_local

    def _make_decomp_endpoint_goals(self,
                                    agent_state: torch.Tensor,
                                    mode_state: torch.Tensor) -> torch.Tensor:
        fallback_goal = self.to_goal(mode_state)
        agent_tokens = agent_state.unsqueeze(1).expand(-1, self.num_modes, -1)
        axis_anchor = self.endpoint_axis_anchor.unsqueeze(0).expand(mode_state.size(0), -1, -1)
        axis_input = torch.cat([mode_state, agent_tokens, fallback_goal, axis_anchor], dim=-1)
        endpoint_axis = axis_anchor + self.to_endpoint_axis(axis_input)
        endpoint_axis = self.topo_mode_endpoint_scale * torch.tanh(endpoint_axis)
        fallback_dir, fallback_ortho = self._build_goal_basis(fallback_goal)
        return fallback_goal + self._compose_axis_delta(endpoint_axis, fallback_dir, fallback_ortho)

    def _make_interaction_decomp_endpoint_goals(self,
                                                data: HeteroData,
                                                scene_enc: Mapping[str, torch.Tensor],
                                                agent_state: torch.Tensor,
                                                mode_state: torch.Tensor) -> torch.Tensor:
        fallback_goal = self.to_goal(mode_state)
        agent_tokens = agent_state.unsqueeze(1).expand(-1, self.num_modes, -1)
        interaction_context = self._extract_interaction_context(data, scene_enc).unsqueeze(1).expand(
            -1, self.num_modes, -1)
        axis_anchor = self.interaction_axis_anchor.unsqueeze(0).expand(mode_state.size(0), -1, -1)
        axis_input = torch.cat([mode_state, agent_tokens, interaction_context, fallback_goal, axis_anchor], dim=-1)
        endpoint_axis = axis_anchor + self.to_interaction_axis(axis_input)
        endpoint_axis = self.topo_mode_endpoint_scale * torch.tanh(endpoint_axis)
        fallback_dir, fallback_ortho = self._build_goal_basis(fallback_goal)
        return fallback_goal + self._compose_axis_delta(endpoint_axis, fallback_dir, fallback_ortho)

    def _make_interaction_cv_endpoint_proposals(self,
                                                data: HeteroData,
                                                scene_enc: Mapping[str, torch.Tensor],
                                                agent_state: torch.Tensor,
                                                mode_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cv_anchor = self._make_cv_anchors(data, mode_state.device, mode_state.dtype)
        fallback_goal = self.to_goal(mode_state)
        cv_goal = cv_anchor[:, -1].unsqueeze(1).expand(-1, self.num_modes, -1)
        cv_goal = cv_goal + self.topo_goal_residual_scale * torch.tanh(fallback_goal)
        agent_tokens = agent_state.unsqueeze(1).expand(-1, self.num_modes, -1)
        interaction_context = self._extract_interaction_context(data, scene_enc).unsqueeze(1).expand(
            -1, self.num_modes, -1)
        axis_anchor = self.interaction_axis_anchor.unsqueeze(0).expand(mode_state.size(0), -1, -1)
        axis_input = torch.cat([mode_state, agent_tokens, interaction_context, cv_goal, axis_anchor], dim=-1)
        endpoint_axis = axis_anchor + self.to_interaction_axis(axis_input)
        endpoint_axis = self.topo_mode_endpoint_scale * torch.tanh(endpoint_axis)
        cv_dir, cv_ortho = self._build_goal_basis(cv_goal)
        endpoint_delta = self._compose_axis_delta(endpoint_axis, cv_dir, cv_ortho)
        goal_local = cv_goal + endpoint_delta

        anchor_base = cv_anchor.unsqueeze(1) + self._make_goal_anchors(endpoint_delta)
        anchor_residual = self.to_anchor_residual(mode_state).view(
            -1, self.num_modes, self.num_future_steps, self.output_dim)
        anchor_local = anchor_base + 0.25 * anchor_residual
        return goal_local, anchor_local

    def _make_attn_endpoint_proposals(self,
                                      data: HeteroData,
                                      scene_enc: Mapping[str, torch.Tensor],
                                      agent_state: torch.Tensor,
                                      mode_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fallback_goal = self.to_goal(mode_state)
        attn_context = self._attend_scene_tokens(data, scene_enc, mode_state)
        agent_tokens = agent_state.unsqueeze(1).expand(-1, self.num_modes, -1)
        endpoint_anchor = self.attn_endpoint_anchor.unsqueeze(0).expand(mode_state.size(0), -1, -1)
        delta_input = torch.cat([mode_state, agent_tokens, attn_context, fallback_goal, endpoint_anchor], dim=-1)
        endpoint_delta = endpoint_anchor + self.to_attn_endpoint_delta(delta_input)
        endpoint_delta = self.topo_mode_endpoint_scale * torch.tanh(endpoint_delta)
        goal_local = fallback_goal + endpoint_delta
        anchor_base = self._make_goal_anchors(goal_local)
        anchor_residual = self.to_anchor_residual(mode_state).view(
            -1, self.num_modes, self.num_future_steps, self.output_dim)
        anchor_local = anchor_base + anchor_residual
        return goal_local, anchor_local

    def _make_topo_query_proposals(self,
                                   data: HeteroData,
                                   scene_enc: Mapping[str, torch.Tensor],
                                   agent_state: torch.Tensor,
                                   mode_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cv_anchor = self._make_cv_anchors(data, mode_state.device, mode_state.dtype)
        cv_goal = cv_anchor[:, -1].unsqueeze(1).expand(-1, self.num_modes, -1)
        fallback_goal = self.to_goal(mode_state)
        agent_context, map_context = self._attend_topo_query_scene(data, scene_enc, mode_state)
        agent_tokens = agent_state.unsqueeze(1).expand(-1, self.num_modes, -1)
        endpoint_anchor = self.topo_query_endpoint_anchor.unsqueeze(0).expand(mode_state.size(0), -1, -1)

        fused_input = torch.cat([
            mode_state,
            agent_tokens,
            agent_context,
            map_context,
            cv_goal,
            fallback_goal,
            endpoint_anchor,
        ], dim=-1)
        proposal_state = self.topo_query_context_fuse(fused_input)
        delta_input = torch.cat([proposal_state, cv_goal, endpoint_anchor], dim=-1)
        endpoint_delta = endpoint_anchor + self.to_topo_query_endpoint_delta(delta_input)
        endpoint_delta = self.topo_mode_endpoint_scale * torch.tanh(endpoint_delta)
        goal_local = cv_goal + endpoint_delta

        anchor_base = cv_anchor.unsqueeze(1) + self._make_goal_anchors(endpoint_delta)
        anchor_residual = self.to_anchor_residual(proposal_state).view(
            -1, self.num_modes, self.num_future_steps, self.output_dim)
        anchor_local = anchor_base + 0.25 * anchor_residual
        return goal_local, anchor_local

    def _attend_topo_query_scene(self,
                                 data: HeteroData,
                                 scene_enc: Mapping[str, torch.Tensor],
                                 mode_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = mode_state.device
        dtype = mode_state.dtype
        agent_feat = scene_enc['x_a'][:, -1].to(device=device, dtype=dtype)
        agent_pos = data['agent']['position'][:, self.num_historical_steps - 1, :2].to(
            device=device, dtype=dtype)
        agent_heading = data['agent']['heading'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)

        map_pos = data['map_polygon']['position'][:, :2].to(device=device, dtype=dtype)
        raw_map_feat = scene_enc.get('x_pl', None)
        if raw_map_feat is not None and map_pos.numel() > 0:
            map_feat = raw_map_feat[:, -1] if raw_map_feat.dim() == 3 else raw_map_feat
            map_feat = map_feat.to(device=device, dtype=dtype)
        else:
            map_feat = agent_feat.new_zeros((0, self.hidden_dim))
            map_pos = agent_pos.new_zeros((0, self.output_dim))

        agent_context = mode_state.new_empty(mode_state.shape)
        map_context = mode_state.new_empty(mode_state.shape)
        if isinstance(data, Batch):
            agent_batch = data['agent']['batch'].to(device=device)
            map_batch = data['map_polygon']['batch'].to(device=device) if map_pos.numel() > 0 else None
            for batch_id in torch.unique(agent_batch, sorted=True):
                agent_idx = torch.nonzero(agent_batch == batch_id, as_tuple=False).flatten()
                if agent_idx.numel() == 0:
                    continue
                agent_context[agent_idx] = self._topo_query_attend_tokens_for_agents(
                    mode_state[agent_idx],
                    agent_pos[agent_idx],
                    agent_heading[agent_idx],
                    agent_pos[agent_idx],
                    agent_feat[agent_idx],
                    self.topo_query_agent_geo_proj,
                    self.topo_query_agent_q,
                    self.topo_query_agent_k,
                    self.topo_query_agent_v,
                )
                if map_batch is not None:
                    map_idx = torch.nonzero(map_batch == batch_id, as_tuple=False).flatten()
                else:
                    map_idx = map_pos.new_zeros((0,), dtype=torch.long)
                if map_idx.numel() > 0:
                    map_context[agent_idx] = self._topo_query_attend_tokens_for_agents(
                        mode_state[agent_idx],
                        agent_pos[agent_idx],
                        agent_heading[agent_idx],
                        map_pos[map_idx],
                        map_feat[map_idx],
                        self.topo_query_map_geo_proj,
                        self.topo_query_map_q,
                        self.topo_query_map_k,
                        self.topo_query_map_v,
                    )
                else:
                    map_context[agent_idx] = mode_state.new_zeros(
                        (agent_idx.numel(), self.num_modes, self.hidden_dim))
        else:
            agent_context = self._topo_query_attend_tokens_for_agents(
                mode_state,
                agent_pos,
                agent_heading,
                agent_pos,
                agent_feat,
                self.topo_query_agent_geo_proj,
                self.topo_query_agent_q,
                self.topo_query_agent_k,
                self.topo_query_agent_v,
            )
            if map_pos.numel() > 0:
                map_context = self._topo_query_attend_tokens_for_agents(
                    mode_state,
                    agent_pos,
                    agent_heading,
                    map_pos,
                    map_feat,
                    self.topo_query_map_geo_proj,
                    self.topo_query_map_q,
                    self.topo_query_map_k,
                    self.topo_query_map_v,
                )
            else:
                map_context = mode_state.new_zeros(mode_state.shape)
        return agent_context, map_context

    def _topo_query_attend_tokens_for_agents(self,
                                             mode_state: torch.Tensor,
                                             agent_pos: torch.Tensor,
                                             agent_heading: torch.Tensor,
                                             token_pos: torch.Tensor,
                                             token_feat: torch.Tensor,
                                             geo_proj: nn.Module,
                                             q_proj: nn.Module,
                                             k_proj: nn.Module,
                                             v_proj: nn.Module) -> torch.Tensor:
        if token_pos.numel() == 0:
            return mode_state.new_zeros(mode_state.shape)
        rel = token_pos.unsqueeze(0) - agent_pos.unsqueeze(1)
        cos = agent_heading.cos().view(-1, 1)
        sin = agent_heading.sin().view(-1, 1)
        local_x = rel[..., 0] * cos + rel[..., 1] * sin
        local_y = -rel[..., 0] * sin + rel[..., 1] * cos
        local_dist = rel.norm(dim=-1) / self.corridor_dist_norm
        token_geo = torch.stack([
            local_x / self.corridor_dist_norm,
            local_y / self.corridor_dist_norm,
            local_dist,
        ], dim=-1)
        token_state = token_feat.unsqueeze(0) + geo_proj(token_geo)
        query = q_proj(mode_state)
        key = k_proj(token_state)
        value = v_proj(token_state)
        score = torch.einsum('nkh,nsh->nks', query, key) / math.sqrt(self.hidden_dim)
        distance_bias = local_dist.unsqueeze(1).clamp_max(4.0)
        forward_bias = F.relu(-local_x / self.corridor_dist_norm).unsqueeze(1).clamp_max(2.0)
        score = score - 0.35 * distance_bias - 0.20 * forward_bias
        weight = torch.softmax(score, dim=-1)
        return torch.einsum('nks,nsh->nkh', weight, value)

    def _attend_scene_tokens(self,
                             data: HeteroData,
                             scene_enc: Mapping[str, torch.Tensor],
                             mode_state: torch.Tensor) -> torch.Tensor:
        device = mode_state.device
        dtype = mode_state.dtype
        agent_feat = scene_enc['x_a'][:, -1].to(device=device, dtype=dtype)
        agent_pos = data['agent']['position'][:, self.num_historical_steps - 1, :2].to(
            device=device, dtype=dtype)
        agent_heading = data['agent']['heading'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)

        map_pos = data['map_polygon']['position'][:, :2].to(device=device, dtype=dtype)
        raw_map_feat = scene_enc.get('x_pl', None)
        if raw_map_feat is not None and map_pos.numel() > 0:
            map_feat = raw_map_feat[:, -1] if raw_map_feat.dim() == 3 else raw_map_feat
            map_feat = map_feat.to(device=device, dtype=dtype)
        else:
            map_feat = agent_feat.new_zeros((0, self.hidden_dim))
            map_pos = agent_pos.new_zeros((0, self.output_dim))

        context = mode_state.new_empty(mode_state.shape)
        if isinstance(data, Batch):
            agent_batch = data['agent']['batch'].to(device=device)
            map_batch = data['map_polygon']['batch'].to(device=device) if map_pos.numel() > 0 else None
            for batch_id in torch.unique(agent_batch, sorted=True):
                agent_idx = torch.nonzero(agent_batch == batch_id, as_tuple=False).flatten()
                if agent_idx.numel() == 0:
                    continue
                if map_batch is not None:
                    map_idx = torch.nonzero(map_batch == batch_id, as_tuple=False).flatten()
                    token_pos = torch.cat([agent_pos[agent_idx], map_pos[map_idx]], dim=0)
                    token_feat = torch.cat([agent_feat[agent_idx], map_feat[map_idx]], dim=0)
                else:
                    token_pos = agent_pos[agent_idx]
                    token_feat = agent_feat[agent_idx]
                context[agent_idx] = self._attend_tokens_for_agents(
                    mode_state[agent_idx],
                    agent_pos[agent_idx],
                    agent_heading[agent_idx],
                    token_pos,
                    token_feat,
                )
        else:
            token_pos = torch.cat([agent_pos, map_pos], dim=0) if map_pos.numel() > 0 else agent_pos
            token_feat = torch.cat([agent_feat, map_feat], dim=0) if map_feat.numel() > 0 else agent_feat
            context = self._attend_tokens_for_agents(mode_state, agent_pos, agent_heading, token_pos, token_feat)
        return context

    def _attend_tokens_for_agents(self,
                                  mode_state: torch.Tensor,
                                  agent_pos: torch.Tensor,
                                  agent_heading: torch.Tensor,
                                  token_pos: torch.Tensor,
                                  token_feat: torch.Tensor) -> torch.Tensor:
        rel = token_pos.unsqueeze(0) - agent_pos.unsqueeze(1)
        cos = agent_heading.cos().view(-1, 1)
        sin = agent_heading.sin().view(-1, 1)
        local_x = rel[..., 0] * cos + rel[..., 1] * sin
        local_y = -rel[..., 0] * sin + rel[..., 1] * cos
        local_dist = rel.norm(dim=-1) / self.corridor_dist_norm
        token_geo = torch.stack([
            local_x / self.corridor_dist_norm,
            local_y / self.corridor_dist_norm,
            local_dist,
        ], dim=-1)
        token_state = token_feat.unsqueeze(0) + self.attn_token_geo_proj(token_geo)
        query = self.attn_q(mode_state)
        key = self.attn_k(token_state)
        value = self.attn_v(token_state)
        score = torch.einsum('nkh,nsh->nks', query, key) / math.sqrt(self.hidden_dim)
        weight = torch.softmax(score, dim=-1)
        attended = torch.einsum('nks,nsh->nkh', weight, value)
        return self.attn_out(attended)

    def _make_cv_anchors(self,
                         data: HeteroData,
                         device: torch.device,
                         dtype: torch.dtype) -> torch.Tensor:
        pos = data['agent']['position'][:, :self.num_historical_steps, :2].to(device=device, dtype=dtype)
        valid = data['agent']['valid_mask'][:, :self.num_historical_steps].to(device=device)
        num_hist = pos.size(1)
        if num_hist <= 1:
            vel_global = pos.new_zeros((pos.size(0), 2))
        else:
            window = min(3, num_hist - 1)
            curr = pos[:, num_hist - window:num_hist]
            prev = pos[:, num_hist - window - 1:num_hist - 1]
            pair_valid = valid[:, num_hist - window:num_hist] & valid[:, num_hist - window - 1:num_hist - 1]
            pair_valid_f = pair_valid.to(dtype=dtype).unsqueeze(-1)
            vel_global = ((curr - prev) * pair_valid_f).sum(dim=1)
            vel_global = vel_global / pair_valid_f.sum(dim=1).clamp_min(1.0)

        heading = data['agent']['heading'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        cos, sin = heading.cos(), heading.sin()
        vel_local = torch.stack([
            vel_global[:, 0] * cos + vel_global[:, 1] * sin,
            -vel_global[:, 0] * sin + vel_global[:, 1] * cos,
        ], dim=-1)
        steps = torch.arange(
            1,
            self.num_future_steps + 1,
            device=device,
            dtype=dtype,
        ).view(1, self.num_future_steps, 1)
        return vel_local.unsqueeze(1) * steps

    def _make_decomp_endpoint_polyline_proposals(self,
                                                 agent_state: torch.Tensor,
                                                 mode_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fallback_goal = self.to_goal(mode_state)
        agent_tokens = agent_state.unsqueeze(1).expand(-1, self.num_modes, -1)
        axis_anchor = self.endpoint_axis_anchor.unsqueeze(0).expand(mode_state.size(0), -1, -1)
        axis_input = torch.cat([mode_state, agent_tokens, fallback_goal, axis_anchor], dim=-1)
        endpoint_axis = axis_anchor + self.to_endpoint_axis(axis_input)
        endpoint_axis = self.topo_mode_endpoint_scale * torch.tanh(endpoint_axis)

        fallback_dir, fallback_ortho = self._build_goal_basis(fallback_goal)
        goal_local = fallback_goal + self._compose_axis_delta(endpoint_axis, fallback_dir, fallback_ortho)

        goal_dir, goal_ortho = self._build_goal_basis(goal_local)
        control_anchor = self.polyline_control_anchor.unsqueeze(0).expand(mode_state.size(0), -1, -1, -1)
        control_input = torch.cat([
            mode_state,
            agent_tokens,
            goal_local,
            endpoint_axis,
            control_anchor.reshape(mode_state.size(0), self.num_modes, -1),
        ], dim=-1)
        control_axis = control_anchor + self.to_polyline_control(control_input).view(
            mode_state.size(0), self.num_modes, self.num_polyline_control_points, 2)
        control_axis = self.topo_polyline_control_scale * torch.tanh(control_axis)
        control_delta = (
            control_axis[..., :1] * goal_dir.unsqueeze(2) +
            control_axis[..., 1:2] * goal_ortho.unsqueeze(2)
        )
        control_alpha = torch.linspace(
            1.0 / (self.num_polyline_control_points + 1),
            self.num_polyline_control_points / (self.num_polyline_control_points + 1),
            self.num_polyline_control_points,
            device=goal_local.device,
            dtype=goal_local.dtype,
        ).view(1, 1, -1, 1)
        control_local = control_alpha * goal_local.unsqueeze(2) + control_delta
        anchor_local = self._make_piecewise_polyline_anchors(goal_local, control_local)
        return goal_local, anchor_local

    def _build_goal_basis(self, goal_local: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        goal_norm = goal_local.norm(dim=-1, keepdim=True)
        default_dir = torch.zeros_like(goal_local)
        default_dir[..., 0] = 1.0
        goal_dir = torch.where(goal_norm > 1e-3, goal_local / goal_norm.clamp_min(1e-3), default_dir)
        goal_ortho = torch.stack([-goal_dir[..., 1], goal_dir[..., 0]], dim=-1)
        return goal_dir, goal_ortho

    def _extract_interaction_context(self,
                                     data: HeteroData,
                                     scene_enc: Mapping[str, torch.Tensor]) -> torch.Tensor:
        agent_feat = scene_enc['x_a'][:, -1]
        device = agent_feat.device
        dtype = agent_feat.dtype
        pos = data['agent']['position'][:, self.num_historical_steps - 1, :2].to(device=device, dtype=dtype)
        heading = data['agent']['heading'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        context = agent_feat.new_zeros(agent_feat.shape)

        if isinstance(data, Batch):
            agent_batch = data['agent']['batch'].to(device=device)
            for batch_id in torch.unique(agent_batch, sorted=True):
                agent_idx = torch.nonzero(agent_batch == batch_id, as_tuple=False).flatten()
                if agent_idx.numel() <= 1:
                    continue
                context[agent_idx] = self._pool_interaction_neighbors(
                    pos[agent_idx], heading[agent_idx], agent_feat[agent_idx])
        else:
            if pos.size(0) > 1:
                context = self._pool_interaction_neighbors(pos, heading, agent_feat)
        return context

    def _pool_interaction_neighbors(self,
                                    pos: torch.Tensor,
                                    heading: torch.Tensor,
                                    feat: torch.Tensor) -> torch.Tensor:
        num_agents = pos.size(0)
        if num_agents <= 1:
            return feat.new_zeros(feat.shape)

        rel = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist = rel.norm(dim=-1)
        dist = dist + torch.eye(num_agents, device=pos.device, dtype=pos.dtype) * 1e6
        k = min(8, num_agents - 1)
        nearest_dist, nearest_idx = dist.topk(k=k, dim=-1, largest=False)
        neigh_feat = feat[nearest_idx]
        neigh_rel = rel.gather(dim=1, index=nearest_idx.unsqueeze(-1).expand(-1, -1, 2))

        cos = heading.cos().unsqueeze(-1)
        sin = heading.sin().unsqueeze(-1)
        rel_x = neigh_rel[..., 0]
        rel_y = neigh_rel[..., 1]
        local_x = rel_x * cos + rel_y * sin
        local_y = -rel_x * sin + rel_y * cos
        neigh_local = torch.stack([local_x, local_y], dim=-1)
        neigh_input = torch.cat([
            neigh_feat,
            neigh_local,
            nearest_dist.unsqueeze(-1) / self.corridor_dist_norm,
        ], dim=-1)
        neigh_token = self.to_interaction_context(neigh_input)
        neigh_weight = torch.softmax(-nearest_dist / 10.0, dim=-1)
        return (neigh_weight.unsqueeze(-1) * neigh_token).sum(dim=1)

    def _compose_axis_delta(self,
                            axis_coeff: torch.Tensor,
                            goal_dir: torch.Tensor,
                            goal_ortho: torch.Tensor) -> torch.Tensor:
        return axis_coeff[..., :1] * goal_dir + axis_coeff[..., 1:2] * goal_ortho

    def _make_piecewise_polyline_anchors(self,
                                         goal_local: torch.Tensor,
                                         control_local: torch.Tensor) -> torch.Tensor:
        start = goal_local.new_zeros((goal_local.size(0), goal_local.size(1), 1, self.output_dim))
        nodes = torch.cat([start, control_local, goal_local.unsqueeze(2)], dim=2)
        num_segments = nodes.size(2) - 1
        node_alpha = torch.linspace(
            0.0, 1.0, num_segments + 1, device=goal_local.device, dtype=goal_local.dtype)
        step_alpha = torch.linspace(
            1.0 / self.num_future_steps, 1.0, self.num_future_steps,
            device=goal_local.device, dtype=goal_local.dtype)
        anchors = []
        for alpha in step_alpha:
            segment_idx = min(int(torch.floor(alpha * num_segments).item()), num_segments - 1)
            left_alpha = node_alpha[segment_idx]
            right_alpha = node_alpha[segment_idx + 1]
            blend = (alpha - left_alpha) / (right_alpha - left_alpha + 1e-6)
            left_node = nodes[:, :, segment_idx]
            right_node = nodes[:, :, segment_idx + 1]
            anchors.append(left_node + blend * (right_node - left_node))
        return torch.stack(anchors, dim=2)

    def _make_corridor_mode_endpoint_proposals(self,
                                               data: HeteroData,
                                               scene_enc: Mapping[str, torch.Tensor],
                                               agent_state: torch.Tensor,
                                               mode_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = mode_state.device
        dtype = mode_state.dtype
        base_goal = self._make_mode_endpoint_goals(agent_state, mode_state)
        fallback_anchor = self._make_goal_anchors(base_goal) + self.to_anchor_residual(mode_state).view(
            -1, self.num_modes, self.num_future_steps, self.output_dim)

        map_pos = data['map_polygon']['position'][:, :2].to(device=device, dtype=dtype)
        if map_pos.numel() == 0:
            return base_goal, fallback_anchor

        map_feat = scene_enc['x_pl'][:, -1].to(device=device, dtype=dtype)
        agent_origin = data['agent']['position'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        agent_heading = data['agent']['heading'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        goal_local = base_goal.new_empty(base_goal.shape)
        anchor_local = fallback_anchor.new_empty(fallback_anchor.shape)

        if isinstance(data, Batch):
            agent_batch = data['agent']['batch'].to(device=device)
            map_batch = data['map_polygon']['batch'].to(device=device)
            for batch_id in torch.unique(agent_batch, sorted=True):
                agent_idx = torch.nonzero(agent_batch == batch_id, as_tuple=False).flatten()
                map_idx = torch.nonzero(map_batch == batch_id, as_tuple=False).flatten()
                if agent_idx.numel() == 0:
                    continue
                if map_idx.numel() == 0:
                    goal_local[agent_idx] = base_goal[agent_idx]
                    anchor_local[agent_idx] = fallback_anchor[agent_idx]
                    continue
                goal_part, anchor_part = self._select_corridor_mode_endpoint_for_agents(
                    mode_state[agent_idx],
                    base_goal[agent_idx],
                    fallback_anchor[agent_idx],
                    agent_origin[agent_idx],
                    agent_heading[agent_idx],
                    map_pos[map_idx],
                    map_feat[map_idx],
                )
                goal_local[agent_idx] = goal_part
                anchor_local[agent_idx] = anchor_part
        else:
            goal_local, anchor_local = self._select_corridor_mode_endpoint_for_agents(
                mode_state,
                base_goal,
                fallback_anchor,
                agent_origin,
                agent_heading,
                map_pos,
                map_feat,
            )
        return goal_local, anchor_local

    def _select_corridor_mode_endpoint_for_agents(self,
                                                  mode_state: torch.Tensor,
                                                  base_goal: torch.Tensor,
                                                  fallback_anchor: torch.Tensor,
                                                  agent_origin: torch.Tensor,
                                                  agent_heading: torch.Tensor,
                                                  map_pos: torch.Tensor,
                                                  map_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_global = self._local_to_global(
            base_goal.unsqueeze(2),
            agent_origin[:, :2],
            agent_heading,
        ).squeeze(2)
        distance = torch.cdist(base_global.reshape(-1, self.output_dim), map_pos).reshape(
            mode_state.size(0), self.num_modes, -1)
        topk = min(8, distance.size(-1))
        nearest_dist, nearest_idx = distance.topk(k=topk, dim=-1, largest=False)
        query_score_all = torch.einsum('nkh,mh->nkm', mode_state, map_feat) / math.sqrt(self.hidden_dim)
        query_score = query_score_all.gather(dim=-1, index=nearest_idx)
        local_score = query_score - self.topo_goal_distance_weight * nearest_dist / self.corridor_dist_norm
        corridor_order = local_score.topk(k=topk, dim=-1, largest=True).indices
        mode_rank = torch.linspace(
            0,
            topk - 1,
            self.num_modes,
            device=mode_state.device,
        ).round().long().view(1, self.num_modes, 1)
        selected = nearest_idx.gather(
            dim=-1,
            index=corridor_order.gather(dim=-1, index=mode_rank.expand(mode_state.size(0), -1, -1)),
        ).squeeze(-1)
        selected_global = map_pos[selected]
        selected_local = self._global_to_local(selected_global, agent_origin[:, :2], agent_heading)
        selected_feat = map_feat[selected]
        corridor_offset = selected_local - base_goal
        delta_input = torch.cat([mode_state, selected_feat, base_goal, corridor_offset], dim=-1)
        goal_delta = self.topo_goal_residual_scale * torch.tanh(self.to_corridor_mode_endpoint_delta(delta_input))
        blend = max(0.0, min(float(self.topo_goal_anchor_blend), 1.0))
        goal_local = base_goal + blend * goal_delta
        anchor_local = fallback_anchor + self._make_goal_anchors(blend * goal_delta)
        return goal_local, anchor_local

    def _make_corridor_goals(self,
                             data: HeteroData,
                             scene_enc: Mapping[str, torch.Tensor],
                             mode_state: torch.Tensor) -> torch.Tensor:
        device = mode_state.device
        dtype = mode_state.dtype
        fallback_goal = self.to_goal(mode_state)
        map_pos = data['map_polygon']['position'][:, :2].to(device=device, dtype=dtype)
        if map_pos.numel() == 0:
            return fallback_goal

        map_feat = scene_enc['x_pl'][:, -1].to(device=device, dtype=dtype)
        agent_origin = data['agent']['position'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        agent_heading = data['agent']['heading'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        goal_local = fallback_goal.new_empty(fallback_goal.shape)

        if isinstance(data, Batch):
            agent_batch = data['agent']['batch'].to(device=device)
            map_batch = data['map_polygon']['batch'].to(device=device)
            for batch_id in torch.unique(agent_batch, sorted=True):
                agent_idx = torch.nonzero(agent_batch == batch_id, as_tuple=False).flatten()
                map_idx = torch.nonzero(map_batch == batch_id, as_tuple=False).flatten()
                if agent_idx.numel() == 0:
                    continue
                if map_idx.numel() == 0:
                    goal_local[agent_idx] = fallback_goal[agent_idx]
                    continue
                goal_local[agent_idx] = self._select_corridor_goals_for_agents(
                    mode_state[agent_idx],
                    fallback_goal[agent_idx],
                    agent_origin[agent_idx],
                    agent_heading[agent_idx],
                    map_pos[map_idx],
                    map_feat[map_idx],
                )
        else:
            goal_local = self._select_corridor_goals_for_agents(
                mode_state,
                fallback_goal,
                agent_origin,
                agent_heading,
                map_pos,
                map_feat,
            )
        return goal_local

    def _make_corridor_residual_goals(self,
                                      data: HeteroData,
                                      scene_enc: Mapping[str, torch.Tensor],
                                      mode_state: torch.Tensor) -> torch.Tensor:
        device = mode_state.device
        dtype = mode_state.dtype
        fallback_goal = self.to_goal(mode_state)
        map_pos = data['map_polygon']['position'][:, :2].to(device=device, dtype=dtype)
        if map_pos.numel() == 0:
            return fallback_goal

        map_feat = scene_enc['x_pl'][:, -1].to(device=device, dtype=dtype)
        agent_origin = data['agent']['position'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        agent_heading = data['agent']['heading'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        goal_local = fallback_goal.new_empty(fallback_goal.shape)

        if isinstance(data, Batch):
            agent_batch = data['agent']['batch'].to(device=device)
            map_batch = data['map_polygon']['batch'].to(device=device)
            for batch_id in torch.unique(agent_batch, sorted=True):
                agent_idx = torch.nonzero(agent_batch == batch_id, as_tuple=False).flatten()
                map_idx = torch.nonzero(map_batch == batch_id, as_tuple=False).flatten()
                if agent_idx.numel() == 0:
                    continue
                if map_idx.numel() == 0:
                    goal_local[agent_idx] = fallback_goal[agent_idx]
                    continue
                goal_local[agent_idx] = self._select_corridor_residual_goals_for_agents(
                    mode_state[agent_idx],
                    fallback_goal[agent_idx],
                    agent_origin[agent_idx],
                    agent_heading[agent_idx],
                    map_pos[map_idx],
                    map_feat[map_idx],
                )
        else:
            goal_local = self._select_corridor_residual_goals_for_agents(
                mode_state,
                fallback_goal,
                agent_origin,
                agent_heading,
                map_pos,
                map_feat,
            )
        return goal_local

    def _make_corridor_query_proposals(self,
                                       data: HeteroData,
                                       scene_enc: Mapping[str, torch.Tensor],
                                       mode_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = mode_state.device
        dtype = mode_state.dtype
        fallback_goal = self.to_goal(mode_state)
        fallback_anchor = self._make_goal_anchors(fallback_goal) + self.to_anchor_residual(mode_state).view(
            -1, self.num_modes, self.num_future_steps, self.output_dim)

        map_pos = data['map_polygon']['position'][:, :2].to(device=device, dtype=dtype)
        if map_pos.numel() == 0:
            return fallback_goal, fallback_anchor

        map_feat = scene_enc['x_pl'][:, -1].to(device=device, dtype=dtype)
        agent_origin = data['agent']['position'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        agent_heading = data['agent']['heading'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        goal_local = fallback_goal.new_empty(fallback_goal.shape)
        anchor_local = fallback_anchor.new_empty(fallback_anchor.shape)

        if isinstance(data, Batch):
            agent_batch = data['agent']['batch'].to(device=device)
            map_batch = data['map_polygon']['batch'].to(device=device)
            for batch_id in torch.unique(agent_batch, sorted=True):
                agent_idx = torch.nonzero(agent_batch == batch_id, as_tuple=False).flatten()
                map_idx = torch.nonzero(map_batch == batch_id, as_tuple=False).flatten()
                if agent_idx.numel() == 0:
                    continue
                if map_idx.numel() == 0:
                    goal_local[agent_idx] = fallback_goal[agent_idx]
                    anchor_local[agent_idx] = fallback_anchor[agent_idx]
                    continue
                goal_part, anchor_part = self._select_corridor_query_for_agents(
                    mode_state[agent_idx],
                    fallback_goal[agent_idx],
                    fallback_anchor[agent_idx],
                    agent_origin[agent_idx],
                    agent_heading[agent_idx],
                    map_pos[map_idx],
                    map_feat[map_idx],
                )
                goal_local[agent_idx] = goal_part
                anchor_local[agent_idx] = anchor_part
        else:
            goal_local, anchor_local = self._select_corridor_query_for_agents(
                mode_state,
                fallback_goal,
                fallback_anchor,
                agent_origin,
                agent_heading,
                map_pos,
                map_feat,
            )
        return goal_local, anchor_local

    def _make_corridor_query_safe_proposals(self,
                                            data: HeteroData,
                                            scene_enc: Mapping[str, torch.Tensor],
                                            mode_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = mode_state.device
        dtype = mode_state.dtype
        fallback_goal = self.to_goal(mode_state)
        fallback_anchor = self._make_goal_anchors(fallback_goal) + self.to_anchor_residual(mode_state).view(
            -1, self.num_modes, self.num_future_steps, self.output_dim)

        map_pos = data['map_polygon']['position'][:, :2].to(device=device, dtype=dtype)
        if map_pos.numel() == 0:
            return fallback_goal, fallback_anchor

        map_feat = scene_enc['x_pl'][:, -1].to(device=device, dtype=dtype)
        agent_origin = data['agent']['position'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        agent_heading = data['agent']['heading'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        goal_local = fallback_goal.new_empty(fallback_goal.shape)
        anchor_local = fallback_anchor.new_empty(fallback_anchor.shape)

        if isinstance(data, Batch):
            agent_batch = data['agent']['batch'].to(device=device)
            map_batch = data['map_polygon']['batch'].to(device=device)
            for batch_id in torch.unique(agent_batch, sorted=True):
                agent_idx = torch.nonzero(agent_batch == batch_id, as_tuple=False).flatten()
                map_idx = torch.nonzero(map_batch == batch_id, as_tuple=False).flatten()
                if agent_idx.numel() == 0:
                    continue
                if map_idx.numel() == 0:
                    goal_local[agent_idx] = fallback_goal[agent_idx]
                    anchor_local[agent_idx] = fallback_anchor[agent_idx]
                    continue
                goal_part, anchor_part = self._select_corridor_query_safe_for_agents(
                    mode_state[agent_idx],
                    fallback_goal[agent_idx],
                    fallback_anchor[agent_idx],
                    agent_origin[agent_idx],
                    agent_heading[agent_idx],
                    map_pos[map_idx],
                    map_feat[map_idx],
                )
                goal_local[agent_idx] = goal_part
                anchor_local[agent_idx] = anchor_part
        else:
            goal_local, anchor_local = self._select_corridor_query_safe_for_agents(
                mode_state,
                fallback_goal,
                fallback_anchor,
                agent_origin,
                agent_heading,
                map_pos,
                map_feat,
            )
        return goal_local, anchor_local

    def _select_corridor_query_safe_for_agents(self,
                                               mode_state: torch.Tensor,
                                               fallback_goal: torch.Tensor,
                                               fallback_anchor: torch.Tensor,
                                               agent_origin: torch.Tensor,
                                               agent_heading: torch.Tensor,
                                               map_pos: torch.Tensor,
                                               map_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fallback_global = self._local_to_global(
            fallback_goal.unsqueeze(2),
            agent_origin[:, :2],
            agent_heading,
        ).squeeze(2)
        distance = torch.cdist(fallback_global.reshape(-1, self.output_dim), map_pos).reshape(
            mode_state.size(0), self.num_modes, -1)
        topk = min(8, distance.size(-1))
        nearest_dist, nearest_idx = distance.topk(k=topk, dim=-1, largest=False)
        query_score_all = torch.einsum('nkh,mh->nkm', mode_state, map_feat) / math.sqrt(self.hidden_dim)
        query_score = query_score_all.gather(dim=-1, index=nearest_idx)
        local_score = query_score - self.topo_goal_distance_weight * nearest_dist / self.corridor_dist_norm
        selected_in_topk = local_score.argmax(dim=-1, keepdim=True)
        selected = nearest_idx.gather(dim=-1, index=selected_in_topk).squeeze(-1)
        selected_global = map_pos[selected]
        selected_local = self._global_to_local(selected_global, agent_origin[:, :2], agent_heading)
        selected_feat = map_feat[selected]
        corridor_offset = selected_local - fallback_goal

        goal_delta_input = torch.cat([mode_state, selected_feat, corridor_offset], dim=-1)
        goal_delta = self.topo_goal_residual_scale * torch.tanh(self.to_corridor_goal_delta(goal_delta_input))
        distance_norm = corridor_offset.norm(dim=-1, keepdim=True) / self.corridor_dist_norm
        mode_progress = torch.linspace(
            0.0,
            1.0,
            self.num_modes,
            device=mode_state.device,
            dtype=mode_state.dtype,
        ).view(1, self.num_modes, 1).expand(mode_state.size(0), -1, -1)
        anchor_delta_input = torch.cat([mode_state, selected_feat, corridor_offset, distance_norm, mode_progress], dim=-1)
        anchor_delta = self.topo_goal_residual_scale * torch.tanh(
            self.to_corridor_anchor_delta(anchor_delta_input).view(
                mode_state.size(0),
                self.num_modes,
                self.num_future_steps,
                self.output_dim,
            )
        )
        blend = max(0.0, min(float(self.topo_goal_anchor_blend), 1.0))
        goal_local = fallback_goal + blend * goal_delta
        goal_path_delta = self._make_goal_anchors(blend * goal_delta)
        anchor_local = fallback_anchor + goal_path_delta + blend * anchor_delta
        return goal_local, anchor_local

    def _select_corridor_query_for_agents(self,
                                          mode_state: torch.Tensor,
                                          fallback_goal: torch.Tensor,
                                          fallback_anchor: torch.Tensor,
                                          agent_origin: torch.Tensor,
                                          agent_heading: torch.Tensor,
                                          map_pos: torch.Tensor,
                                          map_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query_score = torch.einsum('nkh,mh->nkm', mode_state, map_feat) / math.sqrt(self.hidden_dim)
        fallback_global = self._local_to_global(
            fallback_goal.unsqueeze(2),
            agent_origin[:, :2],
            agent_heading,
        ).squeeze(2)
        distance = torch.cdist(fallback_global.reshape(-1, self.output_dim), map_pos).reshape(
            mode_state.size(0), self.num_modes, -1)
        score = query_score - self.topo_goal_distance_weight * distance / self.corridor_dist_norm
        selected = score.argmax(dim=-1)
        selected_global = map_pos[selected]
        selected_local = self._global_to_local(selected_global, agent_origin[:, :2], agent_heading)
        selected_feat = map_feat[selected]
        corridor_offset = selected_local - fallback_goal

        goal_delta_input = torch.cat([mode_state, selected_feat, corridor_offset], dim=-1)
        goal_delta = self.topo_goal_residual_scale * torch.tanh(self.to_corridor_goal_delta(goal_delta_input))
        corridor_goal = selected_local + goal_delta

        distance_norm = corridor_offset.norm(dim=-1, keepdim=True) / self.corridor_dist_norm
        mode_progress = torch.linspace(
            0.0,
            1.0,
            self.num_modes,
            device=mode_state.device,
            dtype=mode_state.dtype,
        ).view(1, self.num_modes, 1).expand(mode_state.size(0), -1, -1)
        anchor_delta_input = torch.cat([mode_state, selected_feat, corridor_offset, distance_norm, mode_progress], dim=-1)
        anchor_delta = self.topo_goal_residual_scale * torch.tanh(
            self.to_corridor_anchor_delta(anchor_delta_input).view(
                mode_state.size(0),
                self.num_modes,
                self.num_future_steps,
                self.output_dim,
            )
        )
        corridor_anchor = self._make_goal_anchors(corridor_goal) + anchor_delta
        blend = max(0.0, min(float(self.topo_goal_anchor_blend), 1.0))
        goal_local = fallback_goal + blend * (corridor_goal - fallback_goal)
        anchor_local = fallback_anchor + blend * (corridor_anchor - fallback_anchor)
        return goal_local, anchor_local

    def _select_corridor_residual_goals_for_agents(self,
                                                   mode_state: torch.Tensor,
                                                   fallback_goal: torch.Tensor,
                                                   agent_origin: torch.Tensor,
                                                   agent_heading: torch.Tensor,
                                                   map_pos: torch.Tensor,
                                                   map_feat: torch.Tensor) -> torch.Tensor:
        query_score = torch.einsum('nkh,mh->nkm', mode_state, map_feat) / math.sqrt(self.hidden_dim)
        fallback_global = self._local_to_global(
            fallback_goal.unsqueeze(2),
            agent_origin[:, :2],
            agent_heading,
        ).squeeze(2)
        distance = torch.cdist(fallback_global.reshape(-1, self.output_dim), map_pos).reshape(
            mode_state.size(0), self.num_modes, -1)
        score = query_score - self.topo_goal_distance_weight * distance / self.corridor_dist_norm
        selected = score.argmax(dim=-1)
        selected_global = map_pos[selected]
        selected_local = self._global_to_local(selected_global, agent_origin[:, :2], agent_heading)
        selected_feat = map_feat[selected]
        corridor_offset = selected_local - fallback_goal
        delta_input = torch.cat([mode_state, selected_feat, corridor_offset], dim=-1)
        delta = self.topo_goal_residual_scale * torch.tanh(self.to_corridor_goal_delta(delta_input))
        blend = max(0.0, min(float(self.topo_goal_anchor_blend), 1.0))
        return fallback_goal + blend * delta

    def _select_corridor_goals_for_agents(self,
                                          mode_state: torch.Tensor,
                                          fallback_goal: torch.Tensor,
                                          agent_origin: torch.Tensor,
                                          agent_heading: torch.Tensor,
                                          map_pos: torch.Tensor,
                                          map_feat: torch.Tensor) -> torch.Tensor:
        query_score = torch.einsum('nkh,mh->nkm', mode_state, map_feat) / math.sqrt(self.hidden_dim)
        fallback_global = self._local_to_global(
            fallback_goal.unsqueeze(2),
            agent_origin[:, :2],
            agent_heading,
        ).squeeze(2)
        distance = torch.cdist(fallback_global.reshape(-1, self.output_dim), map_pos).reshape(
            mode_state.size(0), self.num_modes, -1)
        score = query_score - self.topo_goal_distance_weight * distance / self.corridor_dist_norm
        selected = score.argmax(dim=-1)
        selected_global = map_pos[selected]
        selected_local = self._global_to_local(selected_global, agent_origin[:, :2], agent_heading)
        residual = self.topo_goal_residual_scale * torch.tanh(fallback_goal)
        hard_goal = selected_local + residual
        blend = max(0.0, min(float(self.topo_goal_anchor_blend), 1.0))
        return fallback_goal + blend * (hard_goal - fallback_goal)

    def _extract_corridors(self,
                           data: HeteroData,
                           scene_enc: Mapping[str, torch.Tensor],
                           anchor_local: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = anchor_local.device
        dtype = anchor_local.dtype
        pos_m = data['agent']['position'][:, self.num_historical_steps - 1, :2].to(device=device, dtype=dtype)
        head_m = data['agent']['heading'][:, self.num_historical_steps - 1].to(device=device, dtype=dtype)
        anchor_global = self._local_to_global(anchor_local, pos_m, head_m)

        map_pos = data['map_polygon']['position'][:, :2].to(device=device, dtype=dtype)
        map_feat = scene_enc['x_pl'][:, -1].to(device=device, dtype=dtype)
        n, k, t, _ = anchor_global.shape
        corridor_feat = anchor_local.new_zeros((n, k, t, self.hidden_dim))
        corridor_dist = anchor_local.new_full((n, k, t), self.corridor_dist_norm)
        route_jump = anchor_local.new_zeros((n, k, t))

        if map_pos.numel() == 0:
            return corridor_feat, corridor_dist / self.corridor_dist_norm, route_jump

        if isinstance(data, Batch):
            agent_batch = data['agent']['batch'].to(device=device)
            map_batch = data['map_polygon']['batch'].to(device=device)
            for batch_id in torch.unique(agent_batch, sorted=True):
                agent_idx = torch.nonzero(agent_batch == batch_id, as_tuple=False).flatten()
                map_idx = torch.nonzero(map_batch == batch_id, as_tuple=False).flatten()
                if agent_idx.numel() == 0 or map_idx.numel() == 0:
                    continue
                nearest, dist = self._nearest_map_nodes(anchor_global[agent_idx], map_pos[map_idx])
                nearest_global = map_idx[nearest]
                corridor_feat[agent_idx] = map_feat[nearest_global]
                corridor_dist[agent_idx] = dist
                route_jump[agent_idx] = self._route_jump_rate(data, map_idx, nearest_global)
        else:
            nearest, dist = self._nearest_map_nodes(anchor_global, map_pos)
            corridor_feat = map_feat[nearest]
            corridor_dist = dist
            route_jump = self._route_jump_rate(data, torch.arange(map_pos.size(0), device=device), nearest)

        return corridor_feat, corridor_dist.clamp(max=self.corridor_dist_norm) / self.corridor_dist_norm, route_jump

    def _nearest_map_nodes(self,
                           query_global: torch.Tensor,
                           map_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n, k, t, _ = query_global.shape
        dist = torch.cdist(query_global.reshape(n * k * t, 2), map_pos)
        nearest = dist.argmin(dim=-1)
        nearest_dist = dist.gather(1, nearest.unsqueeze(1)).squeeze(1)
        return nearest.reshape(n, k, t), nearest_dist.reshape(n, k, t)

    def _route_jump_rate(self,
                         data: HeteroData,
                         map_idx: torch.Tensor,
                         nearest_global: torch.Tensor) -> torch.Tensor:
        if nearest_global.size(-1) <= 1:
            return nearest_global.new_zeros(nearest_global.shape, dtype=torch.float)
        edge = data['map_polygon', 'to', 'map_polygon']['edge_index']
        keep = torch.isin(edge[0], map_idx) & torch.isin(edge[1], map_idx)
        edges = set((int(a), int(b)) for a, b in edge[:, keep].detach().cpu().t().tolist())
        edges |= set((b, a) for a, b in edges)
        jumps = nearest_global.new_zeros(nearest_global.shape, dtype=torch.float)
        flat = nearest_global.detach().cpu()
        jump_cpu = torch.zeros_like(flat, dtype=torch.float)
        for i in range(flat.size(0)):
            for j in range(flat.size(1)):
                for step in range(flat.size(2) - 1):
                    a = int(flat[i, j, step])
                    b = int(flat[i, j, step + 1])
                    jump_cpu[i, j, step + 1] = 0.0 if (a == b or (a, b) in edges) else 1.0
        return jump_cpu.to(device=nearest_global.device, dtype=torch.float)

    def _local_to_global(self,
                         local_xy: torch.Tensor,
                         origin: torch.Tensor,
                         theta: torch.Tensor) -> torch.Tensor:
        cos, sin = theta.cos(), theta.sin()
        rot = local_xy.new_zeros((local_xy.size(0), 2, 2))
        rot[:, 0, 0] = cos
        rot[:, 0, 1] = sin
        rot[:, 1, 0] = -sin
        rot[:, 1, 1] = cos
        return torch.matmul(local_xy, rot.unsqueeze(1)) + origin[:, :2].view(-1, 1, 1, 2)

    def _global_to_local(self,
                         global_xy: torch.Tensor,
                         origin: torch.Tensor,
                         theta: torch.Tensor) -> torch.Tensor:
        cos, sin = theta.cos(), theta.sin()
        rot = global_xy.new_zeros((global_xy.size(0), 2, 2))
        rot[:, 0, 0] = cos
        rot[:, 0, 1] = -sin
        rot[:, 1, 0] = sin
        rot[:, 1, 1] = cos
        if global_xy.dim() == 2:
            centered = global_xy - origin[:, :2]
            return torch.matmul(centered.unsqueeze(1), rot).squeeze(1)
        view_shape = [origin.size(0)] + [1] * (global_xy.dim() - 2) + [2]
        rot_shape = [origin.size(0)] + [1] * (global_xy.dim() - 2) + [2, 2]
        centered = global_xy - origin[:, :2].view(*view_shape)
        return torch.matmul(centered.unsqueeze(-2), rot.view(*rot_shape)).squeeze(-2)

    def _heads_from_positions(self,
                              loc_pos: torch.Tensor,
                              scale_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.output_head:
            zero = loc_pos.new_zeros((loc_pos.size(0), loc_pos.size(1), loc_pos.size(2), 1))
            return zero, zero
        start = loc_pos.new_zeros((loc_pos.size(0), loc_pos.size(1), 1, self.output_dim))
        motion = torch.cat([loc_pos[..., :self.output_dim][:, :, :1] - start,
                            loc_pos[..., :self.output_dim][:, :, 1:] -
                            loc_pos[..., :self.output_dim][:, :, :-1]], dim=2)
        loc_head = torch.atan2(motion[..., 1], motion[..., 0]).unsqueeze(-1)
        conc_head = 1.0 / (scale_pos.mean(dim=-1, keepdim=True) + 0.02)
        return loc_head, conc_head
