
# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
# and DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from FTC.util.misc import inverse_sigmoid
from FTC.temporal_deform_attn import DeformAttn


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=1024, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = DeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
    
        # self attention
        src2, _ = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

    
    
class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, T_ in enumerate(spatial_shapes):
            ref = torch.linspace(0.5, T_ - 0.5, T_, dtype=torch.float32, device=device)  # (t,)
            # ref = ref[None] / (valid_ratios[:, None, lvl] * T_)                          # (bs, t)
            ref = ref[None] / (T_)                          # (bs, t)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]          # (N, t, n_levels)
        return reference_points[..., None]                                               # (N, t, n_levels, 1)

    def forward(self, src, temporal_lens, level_start_index, valid_ratios, pos=None, padding_mask=None):
        '''
        src: shape=(bs, t, c)
        temporal_lens: shape=(n_levels). content: [t1, t2, t3, ...]
        level_start_index: shape=(n_levels,). [0, t1, t1+t2, ...]
        valid_ratios: shape=(bs, n_levels).
        '''
        output = src
        # (bs, t, levels, 1)
        reference_points = self.get_reference_points(temporal_lens, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, temporal_lens, level_start_index, padding_mask)
        return output
    
    
    
    
class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        # Encoder Block
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                        dropout, activation,
                                                        num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Level Embedding 
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 1)

        self._reset_parameters()
        
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DeformAttn):
                m._reset_parameters()

        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, T = mask.shape
        valid_T = torch.sum(mask, 1)
        valid_ratio = valid_T.float() / T
        return valid_ratio    # shape=(bs)

    def forward(self, srcs, pos_embeds,masks ,query_embed=None):
        '''
        Params:
            srcs: list of Tensor with shape (bs, c, t)
            masks: list of Tensor with shape (bs, t)
            pos_embeds: list of Tensor with shape (bs, c, t)
            query_embed: list of Tensor with shape (nq, 2c)
        Returns:
            hs: list, per layer output of decoder
            init_reference_out: reference points predicted from query embeddings
            inter_references_out: reference points predicted from each decoder layer
            memory: (bs, c, t), final output of the encoder
        '''

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        temporal_lens = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, t = src.shape
            temporal_lens.append(t)
            # (bs, c, t) => (bs, t, c)
            src = src.transpose(1, 2) 
            pos_embed = pos_embed.transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)

        src_flatten = torch.cat(src_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        temporal_lens = torch.as_tensor(temporal_lens, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((temporal_lens.new_zeros((1, )), temporal_lens.cumsum(0)[:-1]))

        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)   # (bs, nlevels)
        # deformable encoder
        memory = self.encoder(src_flatten, temporal_lens, level_start_index, valid_ratios, 
            lvl_pos_embed_flatten)  # shape=(bs, t, c)
     
        return memory

    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leaky_relu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        
        