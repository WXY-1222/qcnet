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
from itertools import chain
from itertools import compress
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from losses import MixtureNLLLoss
from losses import NLLLoss
from metrics import Brier
from metrics import MR
from metrics import minADE
from metrics import minAHE
from metrics import minFDE
from metrics import minFHE
from modules import QCNetDecoder
from modules import QCNetEncoder
from modules import TopoSSMDecoder

try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object


class QCNet(pl.LightningModule):

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
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_dec_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 enable_topo_ssm_refiner: bool = False,
                 topo_refine_weight: float = 0.1,
                 topo_score_weight: float = 0.1,
                 topo_ssm_layers: int = 1,
                 topo_mamba_d_state: int = 16,
                 topo_mamba_d_conv: int = 4,
                 topo_mamba_expand: int = 2,
                 topo_zero_init: bool = True,
                 topo_corridor_loss_weight: float = 0.0,
                 topo_score_loss_weight: float = 0.0,
                 topo_score_temperature: float = 0.2,
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
                 mamba_lr: float = 0.0,
                 mamba_weight_decay: float = -1.0,
                 aux_goal_loss_weight: float = 0.0,
                 aux_goal_loss_weight_decay: float = 0.0,
                 aux_goal_warmup_epochs: int = 0,
                 topo_endpoint_diversity_loss_weight: float = 0.0,
                 topo_endpoint_diversity_margin: float = 0.12,
                 topo_aux_score: bool = False,
                 topo_aux_score_detach: bool = True,
                 topo_aux_score_only: bool = False,
                 topo_aux_score_loss_weight: float = 0.0,
                 topo_aux_gt_score_loss_weight: float = 0.0,
                 topo_aux_score_mix: float = 0.0,
                 topo_proposal_distill_only: bool = False,
                 topo_motion_distill_only: bool = False,
                 topo_partial_keep_encoder: bool = False,
                 decoder_type: str = 'qcnet',
                 distill_propose_weight: float = 0.0,
                 distill_refine_weight: float = 0.0,
                 distill_score_weight: float = 0.0,
                 distill_rank_weight: float = 0.0,
                 distill_endpoint_weight: float = 0.0,
                 distill_temperature: float = 1.0,
                 distill_warmup_epochs: int = 0,
                 freeze_encoder: bool = False,
                 use_teacher_proposals: bool = False,
                 teacher_proposal_source: str = 'refine',
                 eval_k: int = 6,
                 submission_dir: str = './',
                 submission_file_name: str = 'submission',
                 **kwargs) -> None:
        super(QCNet, self).__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_freq_bands = num_freq_bands
        self.num_map_layers = num_map_layers
        self.num_agent_layers = num_agent_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.pl2pl_radius = pl2pl_radius
        self.time_span = time_span
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_t2m_steps = num_t2m_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.enable_topo_ssm_refiner = enable_topo_ssm_refiner
        self.topo_refine_weight = topo_refine_weight
        self.topo_score_weight = topo_score_weight
        self.topo_ssm_layers = topo_ssm_layers
        self.topo_mamba_d_state = topo_mamba_d_state
        self.topo_mamba_d_conv = topo_mamba_d_conv
        self.topo_mamba_expand = topo_mamba_expand
        self.topo_zero_init = topo_zero_init
        self.topo_corridor_loss_weight = topo_corridor_loss_weight
        self.topo_score_loss_weight = topo_score_loss_weight
        self.topo_score_temperature = topo_score_temperature
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
        self.mamba_lr = mamba_lr
        self.mamba_weight_decay = mamba_weight_decay
        self.aux_goal_loss_weight = aux_goal_loss_weight
        self.aux_goal_loss_weight_decay = aux_goal_loss_weight_decay
        self.aux_goal_warmup_epochs = aux_goal_warmup_epochs
        self.topo_endpoint_diversity_loss_weight = topo_endpoint_diversity_loss_weight
        self.topo_endpoint_diversity_margin = topo_endpoint_diversity_margin
        self.topo_aux_score = topo_aux_score
        self.topo_aux_score_detach = topo_aux_score_detach
        self.topo_aux_score_only = topo_aux_score_only
        self.topo_aux_score_loss_weight = topo_aux_score_loss_weight
        self.topo_aux_gt_score_loss_weight = topo_aux_gt_score_loss_weight
        self.topo_aux_score_mix = topo_aux_score_mix
        self.topo_proposal_distill_only = topo_proposal_distill_only
        self.topo_motion_distill_only = topo_motion_distill_only
        self.topo_partial_keep_encoder = topo_partial_keep_encoder
        self.decoder_type = decoder_type
        self.distill_propose_weight = distill_propose_weight
        self.distill_refine_weight = distill_refine_weight
        self.distill_score_weight = distill_score_weight
        self.distill_rank_weight = distill_rank_weight
        self.distill_endpoint_weight = distill_endpoint_weight
        self.distill_temperature = distill_temperature
        self.distill_warmup_epochs = distill_warmup_epochs
        self.freeze_encoder = freeze_encoder
        self.use_teacher_proposals = use_teacher_proposals
        self.teacher_proposal_source = teacher_proposal_source
        self.teacher_model = None
        self.eval_k = eval_k
        self.submission_dir = submission_dir
        self.submission_file_name = submission_file_name

        self.encoder = QCNetEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        if decoder_type == 'qcnet':
            self.decoder = QCNetDecoder(
                dataset=dataset,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                output_head=output_head,
                num_historical_steps=num_historical_steps,
                num_future_steps=num_future_steps,
                num_modes=num_modes,
                num_recurrent_steps=num_recurrent_steps,
                num_t2m_steps=num_t2m_steps,
                pl2m_radius=pl2m_radius,
                a2m_radius=a2m_radius,
                num_freq_bands=num_freq_bands,
                num_layers=num_dec_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                enable_topo_ssm_refiner=enable_topo_ssm_refiner,
                topo_refine_weight=topo_refine_weight,
                topo_score_weight=topo_score_weight,
                topo_ssm_layers=topo_ssm_layers,
                topo_mamba_d_state=topo_mamba_d_state,
                topo_mamba_d_conv=topo_mamba_d_conv,
                topo_mamba_expand=topo_mamba_expand,
                topo_zero_init=topo_zero_init,
            )
        elif decoder_type == 'topossm':
            self.decoder = TopoSSMDecoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                output_head=output_head,
                num_historical_steps=num_historical_steps,
                num_future_steps=num_future_steps,
                num_modes=num_modes,
                topo_ssm_layers=topo_ssm_layers,
                topo_mamba_d_state=topo_mamba_d_state,
                topo_mamba_d_conv=topo_mamba_d_conv,
                topo_mamba_expand=topo_mamba_expand,
                dropout=dropout,
                topo_proposal_type=topo_proposal_type,
                topo_goal_distance_weight=topo_goal_distance_weight,
                topo_goal_residual_scale=topo_goal_residual_scale,
                topo_goal_anchor_blend=topo_goal_anchor_blend,
                topo_mode_endpoint_scale=topo_mode_endpoint_scale,
                topo_anchor_basis_scale=topo_anchor_basis_scale,
                topo_polyline_control_scale=topo_polyline_control_scale,
                topo_route_slot_longitudinal_scale=topo_route_slot_longitudinal_scale,
                topo_route_slot_lateral_scale=topo_route_slot_lateral_scale,
                topo_route_slot_topk=topo_route_slot_topk,
                topo_route_slot_soft_temperature=topo_route_slot_soft_temperature,
                topo_aux_score=topo_aux_score,
                topo_aux_score_detach=topo_aux_score_detach,
            )
        else:
            raise ValueError(f'{decoder_type} is not a valid decoder_type')

        self.reg_loss = NLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                       reduction='none')

        self.Brier = Brier(max_guesses=eval_k)
        self.minADE = minADE(max_guesses=eval_k)
        self.minAHE = minAHE(max_guesses=eval_k)
        self.minFDE = minFDE(max_guesses=eval_k)
        self.minFHE = minFHE(max_guesses=eval_k)
        self.MR = MR(max_guesses=eval_k)

        self.test_predictions = dict()
        if self.topo_aux_score_only:
            self._freeze_except_topo_aux_score()
        elif self.topo_proposal_distill_only:
            self._freeze_except_topo_proposal()
        elif self.topo_motion_distill_only:
            self._freeze_except_topo_motion()
        else:
            if self.use_teacher_proposals:
                self._freeze_teacher_overridden_proposal()
            if self.freeze_encoder:
                self._freeze_encoder()

    def forward(self, data: HeteroData):
        scene_enc = self.encoder(data)
        proposal_override = self._teacher_proposal_override(data)
        if proposal_override is not None and self.decoder_type != 'topossm':
            raise RuntimeError('use_teacher_proposals is only supported with decoder_type=topossm')
        if proposal_override is not None:
            pred = self.decoder(data, scene_enc, proposal_override=proposal_override)
        else:
            pred = self.decoder(data, scene_enc)
        return pred

    def _freeze_except_topo_aux_score(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith('decoder.to_topo_aux_pi.'))

    def _freeze_except_topo_proposal(self) -> None:
        proposal_prefixes = (
            'decoder.mode_emb.',
            'decoder.query_mlp.',
            'decoder.to_goal.',
            'decoder.mode_endpoint_anchor',
            'decoder.polyline_control_anchor',
            'decoder.route_slot_axis_anchor',
            'decoder.to_mode_endpoint_delta.',
            'decoder.to_corridor_mode_endpoint_delta.',
            'decoder.to_corridor_goal_delta.',
            'decoder.to_corridor_anchor_delta.',
            'decoder.to_corridor_multi_goal_delta.',
            'decoder.to_corridor_multi_control.',
            'decoder.to_route_slot_axis.',
            'decoder.to_route_slot_control.',
            'decoder.to_anchor_residual.',
            'decoder.to_polyline_control_lite.',
            'decoder.to_readout_goal_delta.',
            'decoder.to_readout_polyline_control.',
            'decoder.to_endpoint_axis.',
            'decoder.to_polyline_control.',
            'decoder.corridor_token_proj.',
            'decoder.spatial_ssm.',
            'decoder.to_loc_propose_pos.',
            'decoder.to_scale_propose_pos.',
        )
        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith(proposal_prefixes))
        if self.topo_partial_keep_encoder:
            for param in self.encoder.parameters():
                param.requires_grad_(True)

    def _freeze_except_topo_motion(self) -> None:
        motion_prefixes = (
            'decoder.mode_emb.',
            'decoder.query_mlp.',
            'decoder.to_goal.',
            'decoder.mode_endpoint_anchor',
            'decoder.polyline_control_anchor',
            'decoder.to_mode_endpoint_delta.',
            'decoder.to_corridor_mode_endpoint_delta.',
            'decoder.to_corridor_goal_delta.',
            'decoder.to_corridor_anchor_delta.',
            'decoder.to_corridor_multi_goal_delta.',
            'decoder.to_corridor_multi_control.',
            'decoder.to_anchor_residual.',
            'decoder.to_polyline_control_lite.',
            'decoder.to_readout_goal_delta.',
            'decoder.to_readout_polyline_control.',
            'decoder.to_endpoint_axis.',
            'decoder.to_polyline_control.',
            'decoder.corridor_token_proj.',
            'decoder.spatial_ssm.',
            'decoder.to_loc_propose_pos.',
            'decoder.to_scale_propose_pos.',
            'decoder.rollout_token_proj.',
            'decoder.temporal_ssm.',
            'decoder.to_loc_refine_pos.',
            'decoder.to_scale_refine_pos.',
        )
        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith(motion_prefixes))
        if self.topo_partial_keep_encoder:
            for param in self.encoder.parameters():
                param.requires_grad_(True)

    def _freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad_(False)

    def _freeze_teacher_overridden_proposal(self) -> None:
        for module_name in ('to_goal', 'to_anchor_residual', 'to_corridor_goal_delta', 'to_mode_endpoint_delta'):
            module = getattr(self.decoder, module_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad_(False)
        mode_endpoint_anchor = getattr(self.decoder, 'mode_endpoint_anchor', None)
        if mode_endpoint_anchor is not None:
            mode_endpoint_anchor.requires_grad_(False)
        mode_anchor_basis_offset = getattr(self.decoder, 'mode_anchor_basis_offset', None)
        if mode_anchor_basis_offset is not None:
            mode_anchor_basis_offset.requires_grad_(False)

    def _teacher_proposal_override(self, data: HeteroData):
        if not self.use_teacher_proposals:
            return None
        if self.teacher_model is None:
            raise RuntimeError('use_teacher_proposals requires --distill_teacher_checkpoint')
        source_key = {
            'propose': 'loc_propose_pos',
            'refine': 'loc_refine_pos',
        }.get(self.teacher_proposal_source)
        if source_key is None:
            raise ValueError(f'{self.teacher_proposal_source} is not a valid teacher_proposal_source')
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_pred = self.teacher_model(data)
        return {'loc_propose_pos': teacher_pred[source_key]}

    def training_step(self,
                      data,
                      batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        pred = self(data)
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        topo_corridor_loss, topo_score_loss, topo_aux_score_loss = self._topology_aux_losses(
            pred=pred,
            pi=pi,
            best_mode=best_mode,
            reg_mask=reg_mask,
            cls_mask=cls_mask)
        topo_endpoint_diversity_loss = self._protected_endpoint_diversity_loss(
            pred=pred,
            best_mode=best_mode,
            cls_mask=cls_mask)
        aux_goal_loss = self._matched_mode_aux_goal_loss(
            pred=pred,
            target=gt[..., :self.output_dim],
            best_mode=best_mode,
            cls_mask=cls_mask)
        distill_losses = self._teacher_distill_losses(
            data=data,
            pred=pred,
            pi=pi,
            reg_mask=reg_mask,
            cls_mask=cls_mask)
        self.log('train_reg_loss_propose', reg_loss_propose, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        if topo_corridor_loss is not None:
            self.log('train_topo_corridor_loss', topo_corridor_loss, prog_bar=False, on_step=True, on_epoch=True,
                     batch_size=1)
        if topo_score_loss is not None:
            self.log('train_topo_score_loss', topo_score_loss, prog_bar=False, on_step=True, on_epoch=True,
                     batch_size=1)
        if topo_aux_score_loss is not None:
            self.log('train_topo_aux_score_loss', topo_aux_score_loss, prog_bar=False, on_step=True, on_epoch=True,
                     batch_size=1)
        if topo_endpoint_diversity_loss is not None:
            self.log('train_topo_endpoint_diversity_loss', topo_endpoint_diversity_loss, prog_bar=False, on_step=True,
                     on_epoch=True, batch_size=1)
        if aux_goal_loss is not None:
            self.log('train_aux_goal_loss', aux_goal_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            self.log('train_aux_goal_weight', float(self._aux_goal_loss_scale()), prog_bar=False, on_step=True,
                     on_epoch=False, batch_size=1)
        topo_aux_gt_score_loss = None
        if 'topo_aux_pi' in pred:
            topo_aux_gt_score_loss = F.cross_entropy(pred['topo_aux_pi'], best_mode.detach(), reduction='none')
            topo_aux_gt_score_loss = (
                topo_aux_gt_score_loss * cls_mask.to(dtype=topo_aux_gt_score_loss.dtype)
            ).sum() / cls_mask.sum().clamp_(min=1)
            self.log('train_topo_aux_gt_score_loss', topo_aux_gt_score_loss, prog_bar=False,
                     on_step=True, on_epoch=True, batch_size=1)
        for name, loss_value in distill_losses.items():
            self.log(f'train_{name}', loss_value, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        if self.topo_aux_score_only:
            if topo_aux_score_loss is None and topo_aux_gt_score_loss is None:
                raise RuntimeError('topo_aux_score_only requires decoder output topo_aux_pi')
            aux_loss = 0.0
            if topo_aux_score_loss is not None:
                aux_loss = aux_loss + self.topo_aux_score_loss_weight * topo_aux_score_loss
            if topo_aux_gt_score_loss is not None:
                aux_loss = aux_loss + self.topo_aux_gt_score_loss_weight * topo_aux_gt_score_loss
            distill_scale = self._distill_scale()
            if distill_losses:
                aux_loss = aux_loss + distill_scale * (
                    self.distill_score_weight * distill_losses.get('distill_score_loss', 0.0) +
                    self.distill_rank_weight * distill_losses.get('distill_rank_loss', 0.0))
            return aux_loss
        if self.topo_proposal_distill_only:
            proposal_loss = reg_loss_propose
            if distill_losses:
                proposal_loss = proposal_loss + self._distill_scale() * (
                    self.distill_propose_weight * distill_losses.get('distill_propose_loss', 0.0) +
                    self.distill_endpoint_weight * distill_losses.get('distill_endpoint_loss', 0.0))
            if topo_corridor_loss is not None:
                proposal_loss = proposal_loss + self.topo_corridor_loss_weight * topo_corridor_loss
            if topo_endpoint_diversity_loss is not None:
                proposal_loss = proposal_loss + self.topo_endpoint_diversity_loss_weight * topo_endpoint_diversity_loss
            if aux_goal_loss is not None:
                proposal_loss = proposal_loss + self._aux_goal_loss_scale() * aux_goal_loss
            return proposal_loss
        if self.topo_motion_distill_only:
            motion_loss = reg_loss_propose + reg_loss_refine
            if distill_losses:
                motion_loss = motion_loss + self._distill_scale() * (
                    self.distill_propose_weight * distill_losses.get('distill_propose_loss', 0.0) +
                    self.distill_refine_weight * distill_losses.get('distill_refine_loss', 0.0))
            if topo_corridor_loss is not None:
                motion_loss = motion_loss + self.topo_corridor_loss_weight * topo_corridor_loss
            if topo_endpoint_diversity_loss is not None:
                motion_loss = motion_loss + self.topo_endpoint_diversity_loss_weight * topo_endpoint_diversity_loss
            if aux_goal_loss is not None:
                motion_loss = motion_loss + self._aux_goal_loss_scale() * aux_goal_loss
            return motion_loss
        loss = reg_loss_propose + reg_loss_refine + cls_loss
        if topo_corridor_loss is not None:
            loss = loss + self.topo_corridor_loss_weight * topo_corridor_loss
        if topo_score_loss is not None:
            loss = loss + self.topo_score_loss_weight * topo_score_loss
        if topo_aux_score_loss is not None:
            loss = loss + self.topo_aux_score_loss_weight * topo_aux_score_loss
        if topo_aux_gt_score_loss is not None:
            loss = loss + self.topo_aux_gt_score_loss_weight * topo_aux_gt_score_loss
        if topo_endpoint_diversity_loss is not None:
            loss = loss + self.topo_endpoint_diversity_loss_weight * topo_endpoint_diversity_loss
        if aux_goal_loss is not None:
            loss = loss + self._aux_goal_loss_scale() * aux_goal_loss
        distill_scale = self._distill_scale()
        if distill_losses:
            loss = loss + distill_scale * (
                self.distill_propose_weight * distill_losses.get('distill_propose_loss', 0.0) +
                self.distill_refine_weight * distill_losses.get('distill_refine_loss', 0.0) +
                self.distill_score_weight * distill_losses.get('distill_score_loss', 0.0) +
                self.distill_rank_weight * distill_losses.get('distill_rank_loss', 0.0) +
                self.distill_endpoint_weight * distill_losses.get('distill_endpoint_loss', 0.0))
        return loss

    def validation_step(self,
                        data,
                        batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        pred = self(data)
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        topo_corridor_loss, topo_score_loss, topo_aux_score_loss = self._topology_aux_losses(
            pred=pred,
            pi=pi,
            best_mode=best_mode,
            reg_mask=reg_mask,
            cls_mask=cls_mask)
        self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_cls_loss', cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        if topo_corridor_loss is not None:
            self.log('val_topo_corridor_loss', topo_corridor_loss, prog_bar=True, on_step=False, on_epoch=True,
                     batch_size=1, sync_dist=True)
        if topo_score_loss is not None:
            self.log('val_topo_score_loss', topo_score_loss, prog_bar=False, on_step=False, on_epoch=True,
                     batch_size=1, sync_dist=True)
        if topo_aux_score_loss is not None:
            self.log('val_topo_aux_score_loss', topo_aux_score_loss, prog_bar=False, on_step=False, on_epoch=True,
                     batch_size=1, sync_dist=True)

        if self.dataset in ('argoverse_v2', 'interaction_digir'):
            eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        valid_mask_eval = reg_mask[eval_mask]
        traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
        if not self.output_head:
            traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                     traj_eval[..., :2]], dim=-2)
            motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
            head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
            traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
        pi_eval = F.softmax(self._eval_pi_logits(pred, pi)[eval_mask], dim=-1)
        gt_eval = gt[eval_mask]

        self.Brier.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                          valid_mask=valid_mask_eval)
        self.minADE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.minAHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.minFDE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.minFHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.MR.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                       valid_mask=valid_mask_eval)
        self.log('val_Brier', self.Brier, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minAHE', self.minAHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFHE', self.minFHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

    def test_step(self,
                  data,
                  batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        pred = self(data)
        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = self._eval_pi_logits(pred, pred['pi'])
        if self.dataset in ('argoverse_v2', 'interaction_digir'):
            eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2],
                                 rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)

        traj_eval = traj_eval.cpu().numpy()
        pi_eval = pi_eval.cpu().numpy()
        if self.dataset == 'argoverse_v2':
            eval_id = list(compress(list(chain(*data['agent']['id'])), eval_mask))
            if isinstance(data, Batch):
                for i in range(data.num_graphs):
                    self.test_predictions[data['scenario_id'][i]] = (pi_eval[i], {eval_id[i]: traj_eval[i]})
            else:
                self.test_predictions[data['scenario_id']] = (pi_eval[0], {eval_id[0]: traj_eval[0]})
        elif self.dataset == 'interaction_digir':
            return
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def _protected_endpoint_diversity_loss(self,
                                           pred,
                                           best_mode: torch.Tensor,
                                           cls_mask: torch.Tensor):
        if self.topo_endpoint_diversity_loss_weight <= 0.0:
            return None
        if 'loc_refine_pos' not in pred or pred['loc_refine_pos'].size(1) <= 1:
            return None

        endpoint = pred['loc_refine_pos'][:, :, -1, :self.output_dim]
        num_agents, num_modes, _ = endpoint.shape
        mode_ids = torch.arange(num_modes, device=endpoint.device).view(1, num_modes)
        nonbest = mode_ids != best_mode.detach().view(-1, 1)
        valid = cls_mask.to(dtype=endpoint.dtype)
        valid_den = valid.sum().clamp(min=1.0)
        margin = max(float(self.topo_endpoint_diversity_margin), 1e-6)

        best_endpoint = endpoint[torch.arange(num_agents, device=endpoint.device), best_mode].detach()
        dist_to_best = torch.norm(endpoint - best_endpoint.unsqueeze(1), p=2, dim=-1)
        repel_best = F.relu(margin - dist_to_best) * nonbest.to(dtype=endpoint.dtype)
        repel_best = repel_best.sum(dim=-1) / nonbest.sum(dim=-1).clamp(min=1).to(dtype=endpoint.dtype)

        pair_dist = torch.cdist(endpoint, endpoint, p=2)
        pair_mask = torch.triu(
            torch.ones(num_modes, num_modes, device=endpoint.device, dtype=torch.bool),
            diagonal=1,
        ).unsqueeze(0)
        pair_mask = pair_mask & nonbest.unsqueeze(2) & nonbest.unsqueeze(1)
        pair_loss = F.relu(margin - pair_dist) * pair_mask.to(dtype=endpoint.dtype)
        pair_den = pair_mask.sum(dim=(1, 2)).clamp(min=1).to(dtype=endpoint.dtype)
        pair_loss = pair_loss.sum(dim=(1, 2)) / pair_den

        diversity_loss = 0.5 * (repel_best + pair_loss)
        return (diversity_loss * valid).sum() / valid_den

    def _aux_goal_loss_scale(self) -> float:
        if self.aux_goal_loss_weight <= 0.0 or self.aux_goal_warmup_epochs <= 0:
            return 0.0
        if int(self.current_epoch) >= int(self.aux_goal_warmup_epochs):
            return 0.0
        weight = float(self.aux_goal_loss_weight) - float(self.current_epoch) * float(self.aux_goal_loss_weight_decay)
        return max(weight, 0.0)

    def _matched_mode_aux_goal_loss(self,
                                    pred,
                                    target: torch.Tensor,
                                    best_mode: torch.Tensor,
                                    cls_mask: torch.Tensor):
        if self._aux_goal_loss_scale() <= 0.0:
            return None
        if 'loc_propose_pos' not in pred:
            return None

        pred_endpoint = pred['loc_propose_pos'][:, :, -1, :self.output_dim]
        batch_idx = torch.arange(pred_endpoint.size(0), device=pred_endpoint.device)
        matched_endpoint = pred_endpoint[batch_idx, best_mode]
        target_endpoint = target[:, -1]
        endpoint_loss = F.smooth_l1_loss(matched_endpoint, target_endpoint, reduction='none').sum(dim=-1)
        valid = cls_mask.to(dtype=endpoint_loss.dtype)
        return (endpoint_loss * valid).sum() / valid.sum().clamp(min=1.0)

    def on_test_end(self):
        if self.dataset == 'argoverse_v2':
            ChallengeSubmission(self.test_predictions).to_parquet(
                Path(self.submission_dir) / f'{self.submission_file_name}.parquet')
        elif self.dataset == 'interaction_digir':
            return
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        mamba_core = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                # Mamba core params respond better to a smaller LR and usually should not decay.
                if '.fwd.' in full_param_name or '.bwd.' in full_param_name:
                    mamba_core.add(full_param_name)
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters() if param.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        def make_group(param_names, lr, weight_decay):
            if not param_names:
                return None
            return {
                "params": [param_dict[param_name] for param_name in sorted(param_names)],
                "lr": lr,
                "weight_decay": weight_decay,
            }

        if self.mamba_lr > 0.0:
            base_decay = decay - mamba_core
            base_no_decay = no_decay - mamba_core
            mamba_decay = decay & mamba_core
            mamba_no_decay = no_decay & mamba_core
            mamba_wd = self.mamba_weight_decay if self.mamba_weight_decay >= 0.0 else 0.0
            optim_groups = [
                make_group(base_decay, self.lr, self.weight_decay),
                make_group(base_no_decay, self.lr, 0.0),
                make_group(mamba_decay, self.mamba_lr, mamba_wd),
                make_group(mamba_no_decay, self.mamba_lr, 0.0),
            ]
            optim_groups = [group for group in optim_groups if group is not None]
        else:
            optim_groups = [
                make_group(decay, self.lr, self.weight_decay),
                make_group(no_decay, self.lr, 0.0),
            ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        # The teacher is a frozen training aid, not part of the student model.
        if 'state_dict' in checkpoint:
            checkpoint['state_dict'] = {
                k: v for k, v in checkpoint['state_dict'].items()
                if not k.startswith('teacher_model.')
            }

    def _topology_aux_losses(self,
                             pred,
                             pi: torch.Tensor,
                             best_mode: torch.Tensor,
                             reg_mask: torch.Tensor,
                             cls_mask: torch.Tensor):
        if 'topo_corridor_dist' not in pred:
            return None, None, None
        corridor_dist = pred['topo_corridor_dist']
        batch_idx = torch.arange(corridor_dist.size(0), device=corridor_dist.device)
        mode_dist = corridor_dist[batch_idx, best_mode]
        reg_mask_f = reg_mask.to(dtype=mode_dist.dtype)
        denom = reg_mask_f.sum().clamp_(min=1.0)
        corridor_loss = (mode_dist * reg_mask_f).sum() / denom

        valid_steps = reg_mask_f.unsqueeze(1)
        mode_mean_dist = (corridor_dist * valid_steps).sum(dim=-1) / valid_steps.sum(dim=-1).clamp_(min=1.0)
        target = F.softmax(-mode_mean_dist / max(float(self.topo_score_temperature), 1e-4), dim=-1).detach()
        score_loss = F.kl_div(F.log_softmax(pi, dim=-1), target, reduction='none').sum(dim=-1)
        score_loss = (score_loss * cls_mask.to(dtype=score_loss.dtype)).sum() / cls_mask.sum().clamp_(min=1)
        aux_score_loss = None
        if 'topo_aux_pi' in pred:
            aux_score_loss = F.kl_div(
                F.log_softmax(pred['topo_aux_pi'], dim=-1),
                target,
                reduction='none').sum(dim=-1)
            aux_score_loss = (
                aux_score_loss * cls_mask.to(dtype=aux_score_loss.dtype)).sum() / cls_mask.sum().clamp_(min=1)
        return corridor_loss, score_loss, aux_score_loss

    def _eval_pi_logits(self, pred, pi: torch.Tensor) -> torch.Tensor:
        mix = max(0.0, min(float(self.topo_aux_score_mix), 1.0))
        if mix <= 0.0 or 'topo_aux_pi' not in pred:
            return pi
        return (1.0 - mix) * pi + mix * pred['topo_aux_pi']

    def _distill_scale(self) -> float:
        if self.distill_warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(self.current_epoch + 1) / float(self.distill_warmup_epochs))

    def _teacher_distill_losses(self,
                                data,
                                pred,
                                pi: torch.Tensor,
                                reg_mask: torch.Tensor,
                                cls_mask: torch.Tensor):
        if self.teacher_model is None:
            return {}
        if (self.distill_propose_weight <= 0.0 and self.distill_refine_weight <= 0.0 and
                self.distill_score_weight <= 0.0 and self.distill_rank_weight <= 0.0 and
                self.distill_endpoint_weight <= 0.0):
            return {}

        self.teacher_model.eval()
        with torch.no_grad():
            teacher_pred = self.teacher_model(data)

        valid = reg_mask.to(dtype=pi.dtype)
        valid_count = valid.sum(dim=-1).clamp(min=1.0)
        cls_weight = cls_mask.to(dtype=pi.dtype)
        cls_den = cls_weight.sum().clamp(min=1.0)
        temperature = max(float(self.distill_temperature), 1e-4)
        losses = {}

        def matched_traj_loss(student_pos: torch.Tensor, teacher_pos: torch.Tensor):
            teacher_prob = F.softmax(teacher_pred['pi'].detach() / temperature, dim=-1)
            pair_dist = torch.norm(
                student_pos[:, :, None, :, :self.output_dim] -
                teacher_pos.detach()[:, None, :, :, :self.output_dim],
                p=2,
                dim=-1)
            pair_ade = (pair_dist * valid[:, None, None]).sum(dim=-1) / valid_count[:, None, None]
            min_dist, min_student = pair_ade.min(dim=1)
            loss = (teacher_prob * min_dist).sum(dim=-1)
            loss = (loss * cls_weight).sum() / cls_den
            return loss, min_student, teacher_prob

        if self.distill_endpoint_weight > 0.0:
            teacher_prob = F.softmax(teacher_pred['pi'].detach() / temperature, dim=-1)
            student_endpoint = pred['loc_propose_pos'][:, :, -1, :self.output_dim]
            teacher_endpoint = teacher_pred['loc_refine_pos'].detach()[:, :, -1, :self.output_dim]
            pair_fde = torch.norm(
                student_endpoint[:, :, None] - teacher_endpoint[:, None],
                p=2,
                dim=-1)
            min_fde = pair_fde.min(dim=1).values
            endpoint_loss = (teacher_prob * min_fde).sum(dim=-1)
            endpoint_loss = (endpoint_loss * cls_weight).sum() / cls_den
            losses['distill_endpoint_loss'] = endpoint_loss

        if self.distill_propose_weight > 0.0:
            propose_loss, _, _ = matched_traj_loss(pred['loc_propose_pos'], teacher_pred['loc_propose_pos'])
            losses['distill_propose_loss'] = propose_loss

        if self.distill_refine_weight > 0.0 or self.distill_score_weight > 0.0 or self.distill_rank_weight > 0.0:
            refine_loss, assigned_student, teacher_prob = matched_traj_loss(
                pred['loc_refine_pos'], teacher_pred['loc_refine_pos'])
            score_logits = pred.get('topo_aux_pi', pi) if self.topo_aux_score_only else pi
            if self.distill_refine_weight > 0.0:
                losses['distill_refine_loss'] = refine_loss
            if self.distill_score_weight > 0.0:
                target_prob = pi.new_zeros(pi.shape)
                target_prob.scatter_add_(dim=1, index=assigned_student, src=teacher_prob)
                target_prob = target_prob / target_prob.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                score_loss = F.kl_div(
                    F.log_softmax(score_logits / temperature, dim=-1),
                    target_prob.detach(),
                    reduction='none').sum(dim=-1) * (temperature ** 2)
                score_loss = (score_loss * cls_weight).sum() / cls_den
                losses['distill_score_loss'] = score_loss
            if self.distill_rank_weight > 0.0:
                teacher_top = teacher_prob.argmax(dim=-1, keepdim=True)
                target_student = assigned_student.gather(dim=1, index=teacher_top).squeeze(1).detach()
                rank_loss = F.cross_entropy(score_logits, target_student, reduction='none')
                rank_loss = (rank_loss * cls_weight).sum() / cls_den
                losses['distill_rank_loss'] = rank_loss
        return losses

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('QCNet')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--output_head', action='store_true')
        parser.add_argument('--num_historical_steps', type=int, required=True)
        parser.add_argument('--num_future_steps', type=int, required=True)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_recurrent_steps', type=int, required=True)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--num_dec_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--pl2pl_radius', type=float, required=True)
        parser.add_argument('--time_span', type=int, default=None)
        parser.add_argument('--pl2a_radius', type=float, required=True)
        parser.add_argument('--a2a_radius', type=float, required=True)
        parser.add_argument('--num_t2m_steps', type=int, default=None)
        parser.add_argument('--pl2m_radius', type=float, required=True)
        parser.add_argument('--a2m_radius', type=float, required=True)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--enable_topo_ssm_refiner', action='store_true')
        parser.add_argument('--topo_refine_weight', type=float, default=0.1)
        parser.add_argument('--topo_score_weight', type=float, default=0.1)
        parser.add_argument('--topo_ssm_layers', type=int, default=1)
        parser.add_argument('--topo_mamba_d_state', type=int, default=16)
        parser.add_argument('--topo_mamba_d_conv', type=int, default=4)
        parser.add_argument('--topo_mamba_expand', type=int, default=2)
        parser.add_argument('--topo_zero_init', type=bool, default=True)
        parser.add_argument('--topo_corridor_loss_weight', type=float, default=0.0)
        parser.add_argument('--topo_score_loss_weight', type=float, default=0.0)
        parser.add_argument('--topo_score_temperature', type=float, default=0.2)
        parser.add_argument('--topo_proposal_type', type=str, default='goal_mlp',
                            choices=['goal_mlp', 'mode_endpoint', 'corridor_mode_endpoint', 'corridor_goal',
                                     'corridor_residual', 'corridor_query', 'corridor_query_safe',
                                     'decomp_endpoint', 'decomp_endpoint_polyline',
                                     'mode_endpoint_anchorbasis', 'mode_endpoint_polyline_readout',
                                     'mode_endpoint_polyline_lite', 'corridor_multi_anchor',
                                     'route_slot_polyline', 'soft_route_slot_polyline',
                                     'interaction_decomp_endpoint', 'interaction_cv_endpoint'])
        parser.add_argument('--topo_goal_distance_weight', type=float, default=0.05)
        parser.add_argument('--topo_goal_residual_scale', type=float, default=0.25)
        parser.add_argument('--topo_goal_anchor_blend', type=float, default=1.0)
        parser.add_argument('--topo_mode_endpoint_scale', type=float, default=0.08)
        parser.add_argument('--topo_anchor_basis_scale', type=float, default=0.20)
        parser.add_argument('--topo_polyline_control_scale', type=float, default=0.12)
        parser.add_argument('--topo_route_slot_longitudinal_scale', type=float, default=0.20)
        parser.add_argument('--topo_route_slot_lateral_scale', type=float, default=0.12)
        parser.add_argument('--topo_route_slot_topk', type=int, default=12)
        parser.add_argument('--topo_route_slot_soft_temperature', type=float, default=0.35)
        parser.add_argument('--mamba_lr', type=float, default=0.0,
                            help='Optional lower learning rate for parameters inside BidirectionalMambaBlock.fwd/bwd.')
        parser.add_argument('--mamba_weight_decay', type=float, default=-1.0,
                            help='Optional weight decay for Mamba core params; negative means use zero decay.')
        parser.add_argument('--aux_goal_loss_weight', type=float, default=0.0,
                            help='Warmup-only matched-mode endpoint loss weight on proposal endpoints.')
        parser.add_argument('--aux_goal_loss_weight_decay', type=float, default=0.0,
                            help='Per-epoch decay for the warmup-only matched-mode endpoint loss.')
        parser.add_argument('--aux_goal_warmup_epochs', type=int, default=0,
                            help='Number of early epochs where the matched-mode endpoint aux loss is active.')
        parser.add_argument('--topo_endpoint_diversity_loss_weight', type=float, default=0.0)
        parser.add_argument('--topo_endpoint_diversity_margin', type=float, default=0.12)
        parser.add_argument('--topo_aux_score', action='store_true')
        parser.add_argument('--topo_aux_score_detach', type=bool, default=True)
        parser.add_argument('--topo_aux_score_only', action='store_true')
        parser.add_argument('--topo_aux_score_loss_weight', type=float, default=0.0)
        parser.add_argument('--topo_aux_gt_score_loss_weight', type=float, default=0.0)
        parser.add_argument('--topo_aux_score_mix', type=float, default=0.0)
        parser.add_argument('--topo_proposal_distill_only', action='store_true')
        parser.add_argument('--topo_motion_distill_only', action='store_true')
        parser.add_argument('--topo_partial_keep_encoder', action='store_true',
                            help='Keep encoder trainable when using partial TopoSSM freeze modes.')
        parser.add_argument('--decoder_type', type=str, default='qcnet', choices=['qcnet', 'topossm'])
        parser.add_argument('--distill_propose_weight', type=float, default=0.0)
        parser.add_argument('--distill_refine_weight', type=float, default=0.0)
        parser.add_argument('--distill_score_weight', type=float, default=0.0)
        parser.add_argument('--distill_rank_weight', type=float, default=0.0)
        parser.add_argument('--distill_endpoint_weight', type=float, default=0.0)
        parser.add_argument('--distill_temperature', type=float, default=1.0)
        parser.add_argument('--distill_warmup_epochs', type=int, default=0)
        parser.add_argument('--freeze_encoder', action='store_true')
        parser.add_argument('--use_teacher_proposals', action='store_true')
        parser.add_argument('--teacher_proposal_source', type=str, default='refine', choices=['propose', 'refine'])
        parser.add_argument('--eval_k', type=int, default=6)
        parser.add_argument('--submission_dir', type=str, default='./')
        parser.add_argument('--submission_file_name', type=str, default='submission')
        return parent_parser
