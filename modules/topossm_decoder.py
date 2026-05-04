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
from typing import Dict, Mapping, Tuple

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
        self.topo_aux_score = topo_aux_score
        self.topo_aux_score_detach = topo_aux_score_detach
        self.corridor_dist_norm = corridor_dist_norm
        if topo_proposal_type not in ('goal_mlp', 'corridor_goal', 'corridor_residual'):
            raise ValueError(f'{topo_proposal_type} is not a valid topo_proposal_type')

        self.mode_emb = nn.Embedding(num_modes, hidden_dim)
        self.query_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.to_goal = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        if topo_proposal_type == 'corridor_residual':
            self.to_corridor_goal_delta = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2 + output_dim),
                nn.Linear(hidden_dim * 2 + output_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        self.to_anchor_residual = MLPLayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_future_steps * output_dim,
        )

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
        if topo_proposal_type == 'corridor_residual':
            nn.init.zeros_(self.to_corridor_goal_delta[-1].weight)
            nn.init.zeros_(self.to_corridor_goal_delta[-1].bias)

    def forward(self,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        agent_state = scene_enc['x_a'][:, -1]
        mode_state = agent_state.unsqueeze(1) + self.mode_emb.weight.unsqueeze(0)
        mode_state = self.query_mlp(mode_state)

        if self.topo_proposal_type == 'corridor_goal':
            goal_local = self._make_corridor_goals(data, scene_enc, mode_state)
        elif self.topo_proposal_type == 'corridor_residual':
            goal_local = self._make_corridor_residual_goals(data, scene_enc, mode_state)
        else:
            goal_local = self.to_goal(mode_state)
        anchor_base = self._make_goal_anchors(goal_local)
        anchor_residual = self.to_anchor_residual(mode_state).view(
            -1, self.num_modes, self.num_future_steps, self.output_dim)
        anchor_local = anchor_base + anchor_residual

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
        centered = global_xy - origin[:, :2].view(-1, 1, 2)
        return torch.matmul(centered, rot)

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
