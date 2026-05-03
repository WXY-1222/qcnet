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
from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse

from layers import AttentionLayer
from layers import FourierEmbedding
from layers import MLPLayer
from utils import angle_between_2d_vectors
from utils import bipartite_dense_to_sparse
from utils import weight_init
from utils import wrap_angle

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class BidirectionalMambaBlock(nn.Module):
    """Bidirectional residual wrapper around the official Mamba layer."""

    def __init__(self,
                 hidden_dim: int,
                 d_state: int,
                 d_conv: int,
                 expand: int,
                 dropout: float) -> None:
        super(BidirectionalMambaBlock, self).__init__()
        if Mamba is None:
            raise ImportError('TopologySSMRefiner requires mamba-ssm when --enable_topo_ssm_refiner is used')
        self.fwd_norm = nn.LayerNorm(hidden_dim)
        self.bwd_norm = nn.LayerNorm(hidden_dim)
        self.fwd = Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mix = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_fwd = self.fwd(self.fwd_norm(x))
        y_bwd = torch.flip(self.bwd(self.bwd_norm(torch.flip(x, dims=[1]))), dims=[1])
        x = x + self.dropout(self.mix(torch.cat([y_fwd, y_bwd], dim=-1)))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class TopologySSMRefiner(nn.Module):
    """
    Bind QCNet refined proposals to KG/map corridors, then apply bidirectional
    spatial-temporal Mamba refiners to produce a small trajectory residual and
    topology-aware score bias.
    """

    def __init__(self,
                 hidden_dim: int,
                 output_dim: int,
                 num_future_steps: int,
                 num_layers: int,
                 d_state: int,
                 d_conv: int,
                 expand: int,
                 dropout: float,
                 zero_init: bool) -> None:
        super(TopologySSMRefiner, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_future_steps = num_future_steps
        token_dim = hidden_dim * 2 + output_dim * 2 + 1
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
        )
        self.corridor_ssm = nn.ModuleList([
            BidirectionalMambaBlock(hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.temporal_ssm = nn.ModuleList([
            BidirectionalMambaBlock(hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.zero_init = zero_init

    def reset_output(self) -> None:
        if not self.zero_init:
            return
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def forward(self,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor],
                loc_refine_pos: torch.Tensor,
                mode_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        corridor_feat, corridor_dist = self._bind_proposals_to_corridors(data, scene_enc, loc_refine_pos)
        motion = loc_refine_pos.new_zeros(loc_refine_pos.shape)
        motion[:, :, 1:] = loc_refine_pos[:, :, 1:] - loc_refine_pos[:, :, :-1]
        mode_feat = mode_state.unsqueeze(2).expand(-1, -1, self.num_future_steps, -1)
        token = torch.cat([
            loc_refine_pos[..., :self.output_dim],
            motion[..., :self.output_dim],
            corridor_feat,
            mode_feat,
            corridor_dist.unsqueeze(-1),
        ], dim=-1)
        n, k, t, _ = token.shape
        x = self.token_proj(token).reshape(n * k, t, self.hidden_dim)
        for block in self.corridor_ssm:
            x = block(x)
        for block in self.temporal_ssm:
            x = block(x)
        delta = self.delta_head(x).reshape(n, k, t, self.output_dim)
        score_bias = self.score_head(x.mean(dim=1)).reshape(n, k)
        return {
            'delta': delta,
            'score_bias': score_bias,
            'corridor_dist': corridor_dist,
        }

    def _bind_proposals_to_corridors(self,
                                     data: HeteroData,
                                     scene_enc: Mapping[str, torch.Tensor],
                                     loc_refine_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = loc_refine_pos.device
        dtype = loc_refine_pos.dtype
        pos_m = data['agent']['position'][:, -self.num_future_steps - 1, :2].to(device=device, dtype=dtype)
        head_m = data['agent']['heading'][:, -self.num_future_steps - 1].to(device=device, dtype=dtype)
        cos, sin = head_m.cos(), head_m.sin()
        rot = loc_refine_pos.new_zeros((loc_refine_pos.size(0), 2, 2))
        rot[:, 0, 0] = cos
        rot[:, 0, 1] = sin
        rot[:, 1, 0] = -sin
        rot[:, 1, 1] = cos
        pred_global = torch.matmul(loc_refine_pos[..., :2], rot.unsqueeze(1)) + pos_m[:, :2].view(-1, 1, 1, 2)

        map_pos = data['map_polygon']['position'][:, :2].to(device=device, dtype=dtype)
        x_pl = scene_enc['x_pl'][:, -1].to(device=device, dtype=dtype)
        n, k, t, _ = pred_global.shape
        corridor_feat = loc_refine_pos.new_zeros((n, k, t, self.hidden_dim))
        corridor_dist = loc_refine_pos.new_zeros((n, k, t))

        if isinstance(data, Batch):
            agent_batch = data['agent']['batch'].to(device=device)
            map_batch = data['map_polygon']['batch'].to(device=device)
            batch_ids = torch.unique(agent_batch)
            for batch_id in batch_ids:
                agent_idx = torch.nonzero(agent_batch == batch_id, as_tuple=False).flatten()
                map_idx = torch.nonzero(map_batch == batch_id, as_tuple=False).flatten()
                if agent_idx.numel() == 0 or map_idx.numel() == 0:
                    continue
                query = pred_global[agent_idx].reshape(-1, 2)
                dist = torch.cdist(query, map_pos[map_idx])
                nearest_local = dist.argmin(dim=-1)
                nearest = map_idx[nearest_local].reshape(agent_idx.numel(), k, t)
                corridor_feat[agent_idx] = x_pl[nearest]
                corridor_dist[agent_idx] = dist.gather(1, nearest_local.unsqueeze(1)).squeeze(1).reshape(
                    agent_idx.numel(), k, t)
        else:
            query = pred_global.reshape(-1, 2)
            dist = torch.cdist(query, map_pos)
            nearest = dist.argmin(dim=-1)
            corridor_feat = x_pl[nearest].reshape(n, k, t, self.hidden_dim)
            corridor_dist = dist.gather(1, nearest.unsqueeze(1)).squeeze(1).reshape(n, k, t)

        return corridor_feat, corridor_dist.clamp(max=50.0) / 50.0


class QCNetDecoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 enable_topo_ssm_refiner: bool = False,
                 topo_refine_weight: float = 0.1,
                 topo_score_weight: float = 0.1,
                 topo_ssm_layers: int = 1,
                 topo_mamba_d_state: int = 16,
                 topo_mamba_d_conv: int = 4,
                 topo_mamba_expand: int = 2,
                 topo_zero_init: bool = True) -> None:
        super(QCNetDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_t2m_steps = num_t2m_steps if num_t2m_steps is not None else num_historical_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.enable_topo_ssm_refiner = enable_topo_ssm_refiner
        self.topo_refine_weight = topo_refine_weight
        self.topo_score_weight = topo_score_weight

        input_dim_r_t = 4
        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3

        self.mode_emb = nn.Embedding(num_modes, hidden_dim)
        self.r_t2m_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2m_emb = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2m_emb = FourierEmbedding(input_dim=input_dim_r_a2m, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.y_emb = FourierEmbedding(input_dim=output_dim + output_head, hidden_dim=hidden_dim,
                                      num_freq_bands=num_freq_bands)
        self.traj_emb = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                               batch_first=False, dropout=0.0, bidirectional=False)
        self.traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.t2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_propose_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                     dropout=dropout, bipartite=False, has_pos_emb=False)
        self.t2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_refine_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                    dropout=dropout, bipartite=False, has_pos_emb=False)
        self.to_loc_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=num_future_steps * output_dim // num_recurrent_steps)
        self.to_scale_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                             output_dim=num_future_steps * output_dim // num_recurrent_steps)
        self.to_loc_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                          output_dim=num_future_steps * output_dim)
        self.to_scale_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            output_dim=num_future_steps * output_dim)
        if output_head:
            self.to_loc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps // num_recurrent_steps)
            self.to_conc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                 output_dim=num_future_steps // num_recurrent_steps)
            self.to_loc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_future_steps)
            self.to_conc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps)
        else:
            self.to_loc_propose_head = None
            self.to_conc_propose_head = None
            self.to_loc_refine_head = None
            self.to_conc_refine_head = None
        self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        self.topo_refiner = TopologySSMRefiner(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_future_steps=num_future_steps,
            num_layers=topo_ssm_layers,
            d_state=topo_mamba_d_state,
            d_conv=topo_mamba_d_conv,
            expand=topo_mamba_expand,
            dropout=dropout,
            zero_init=topo_zero_init,
        ) if enable_topo_ssm_refiner else None
        self.apply(weight_init)
        if self.topo_refiner is not None:
            self.topo_refiner.reset_output()

    def forward(self,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pos_m = data['agent']['position'][:, self.num_historical_steps - 1, :self.input_dim]
        head_m = data['agent']['heading'][:, self.num_historical_steps - 1]
        head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1)

        x_t = scene_enc['x_a'].reshape(-1, self.hidden_dim)
        x_pl = scene_enc['x_pl'][:, self.num_historical_steps - 1].repeat(self.num_modes, 1)
        x_a = scene_enc['x_a'][:, -1].repeat(self.num_modes, 1)
        m = self.mode_emb.weight.repeat(scene_enc['x_a'].size(0), 1)

        mask_src = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True).repeat(1, self.num_modes)

        pos_t = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim)
        head_t = data['agent']['heading'][:, :self.num_historical_steps].reshape(-1)
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst[:, -1:].unsqueeze(1))
        rel_pos_t2m = pos_t[edge_index_t2m[0]] - pos_m[edge_index_t2m[1]]
        rel_head_t2m = wrap_angle(head_t[edge_index_t2m[0]] - head_m[edge_index_t2m[1]])
        r_t2m = torch.stack(
            [torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m[1]], nbr_vector=rel_pos_t2m[:, :2]),
             rel_head_t2m,
             (edge_index_t2m[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)
        r_t2m = self.r_t2m_emb(continuous_inputs=r_t2m, categorical_embs=None)
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1))
        r_t2m = r_t2m.repeat_interleave(repeats=self.num_modes, dim=0)

        pos_pl = data['map_polygon']['position'][:, :self.input_dim]
        orient_pl = data['map_polygon']['orientation']
        edge_index_pl2m = radius(
            x=pos_m[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2m_radius,
            batch_x=data['agent']['batch'] if isinstance(data, Batch) else None,
            batch_y=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
            max_num_neighbors=300)
        edge_index_pl2m = edge_index_pl2m[:, mask_dst[edge_index_pl2m[1], 0]]
        rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]]
        rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]])
        r_pl2m = torch.stack(
            [torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]),
             rel_orient_pl2m], dim=-1)
        r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m, categorical_embs=None)
        edge_index_pl2m = torch.cat([edge_index_pl2m + i * edge_index_pl2m.new_tensor(
            [[data['map_polygon']['num_nodes']], [data['agent']['num_nodes']]]) for i in range(self.num_modes)], dim=1)
        r_pl2m = r_pl2m.repeat(self.num_modes, 1)

        edge_index_a2m = radius_graph(
            x=pos_m[:, :2],
            r=self.a2m_radius,
            batch=data['agent']['batch'] if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=300)
        edge_index_a2m = edge_index_a2m[:, mask_src[:, -1][edge_index_a2m[0]] & mask_dst[edge_index_a2m[1], 0]]
        rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]]
        rel_head_a2m = wrap_angle(head_m[edge_index_a2m[0]] - head_m[edge_index_a2m[1]])
        r_a2m = torch.stack(
            [torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2m[1]], nbr_vector=rel_pos_a2m[:, :2]),
             rel_head_a2m], dim=-1)
        r_a2m = self.r_a2m_emb(continuous_inputs=r_a2m, categorical_embs=None)
        edge_index_a2m = torch.cat(
            [edge_index_a2m + i * edge_index_a2m.new_tensor([data['agent']['num_nodes']]) for i in
             range(self.num_modes)], dim=1)
        r_a2m = r_a2m.repeat(self.num_modes, 1)

        edge_index_m2m = dense_to_sparse(mask_dst.unsqueeze(2) & mask_dst.unsqueeze(1))[0]

        locs_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        scales_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        locs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        concs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        for t in range(self.num_recurrent_steps):
            for i in range(self.num_layers):
                m = m.reshape(-1, self.hidden_dim)
                m = self.t2m_propose_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
                m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                m = self.pl2m_propose_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                m = self.a2m_propose_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
                m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.m2m_propose_attn_layer(m, None, edge_index_m2m)
            m = m.reshape(-1, self.num_modes, self.hidden_dim)
            locs_propose_pos[t] = self.to_loc_propose_pos(m)
            scales_propose_pos[t] = self.to_scale_propose_pos(m)
            if self.output_head:
                locs_propose_head[t] = self.to_loc_propose_head(m)
                concs_propose_head[t] = self.to_conc_propose_head(m)
        loc_propose_pos = torch.cumsum(
            torch.cat(locs_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
            dim=-2)
        scale_propose_pos = torch.cumsum(
            F.elu_(
                torch.cat(scales_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
                alpha=1.0) +
            1.0,
            dim=-2) + 0.1
        if self.output_head:
            loc_propose_head = torch.cumsum(torch.tanh(torch.cat(locs_propose_head, dim=-1).unsqueeze(-1)) * math.pi,
                                            dim=-2)
            conc_propose_head = 1.0 / (torch.cumsum(F.elu_(torch.cat(concs_propose_head, dim=-1).unsqueeze(-1)) + 1.0,
                                                    dim=-2) + 0.02)
            m = self.y_emb(torch.cat([loc_propose_pos.detach(),
                                      wrap_angle(loc_propose_head.detach())], dim=-1).view(-1, self.output_dim + 1))
        else:
            loc_propose_head = loc_propose_pos.new_zeros((loc_propose_pos.size(0), self.num_modes,
                                                          self.num_future_steps, 1))
            conc_propose_head = scale_propose_pos.new_zeros((scale_propose_pos.size(0), self.num_modes,
                                                             self.num_future_steps, 1))
            m = self.y_emb(loc_propose_pos.detach().view(-1, self.output_dim))
        m = m.reshape(-1, self.num_future_steps, self.hidden_dim).transpose(0, 1)
        m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0)
        for i in range(self.num_layers):
            m = self.t2m_refine_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
            m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.pl2m_refine_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
            m = self.a2m_refine_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
            m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        m = self.m2m_refine_attn_layer(m, None, edge_index_m2m)
        m = m.reshape(-1, self.num_modes, self.hidden_dim)
        loc_refine_pos = self.to_loc_refine_pos(m).view(-1, self.num_modes, self.num_future_steps, self.output_dim)
        loc_refine_pos = loc_refine_pos + loc_propose_pos.detach()
        scale_refine_pos = F.elu_(
            self.to_scale_refine_pos(m).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
            alpha=1.0) + 1.0 + 0.1
        if self.output_head:
            loc_refine_head = torch.tanh(self.to_loc_refine_head(m).unsqueeze(-1)) * math.pi
            loc_refine_head = loc_refine_head + loc_propose_head.detach()
            conc_refine_head = 1.0 / (F.elu_(self.to_conc_refine_head(m).unsqueeze(-1)) + 1.0 + 0.02)
        else:
            loc_refine_head = loc_refine_pos.new_zeros((loc_refine_pos.size(0), self.num_modes, self.num_future_steps,
                                                        1))
            conc_refine_head = scale_refine_pos.new_zeros((scale_refine_pos.size(0), self.num_modes,
                                                           self.num_future_steps, 1))
        pi = self.to_pi(m).squeeze(-1)
        topo_out = None
        if self.topo_refiner is not None:
            topo_out = self.topo_refiner(data=data, scene_enc=scene_enc, loc_refine_pos=loc_refine_pos, mode_state=m)
            loc_refine_pos = loc_refine_pos + self.topo_refine_weight * topo_out['delta']
            pi = pi + self.topo_score_weight * topo_out['score_bias']

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
        }
        if topo_out is not None:
            out['topo_delta'] = topo_out['delta']
            out['topo_score_bias'] = topo_out['score_bias']
            out['topo_corridor_dist'] = topo_out['corridor_dist']
        return out
