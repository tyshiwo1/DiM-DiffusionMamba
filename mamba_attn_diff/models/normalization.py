from typing import Any, Dict, Optional, Tuple
import numbers

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero

from copy import deepcopy

from mamba_ssm import Mamba


import math 
import einops
from functools import partial
from torch import Tensor
from typing import Optional

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class FusedAddRMSNorm(nn.Module):
    def __init__(self, dim, rms_norm=True, residual_in_fp32=True, drop_path=0., norm_epsilon=1e-5, fused_add_norm=True, prenorm=True, **kwargs, ):
        super().__init__()
        self.dim=dim
        self.rms_norm = rms_norm
        self.norm = nn.LayerNorm(dim) if not rms_norm else RMSNorm(dim, eps=norm_epsilon)
        self.residual_in_fp32 = residual_in_fp32
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.fused_add_norm = fused_add_norm & ((torch.__version__ >= "2.1.0") | (torch.__version__ < "2.0.0"))
        self.prenorm = prenorm
    
    def forward(self, hidden_states, pre_norm_x=None, **kwargs):
        residual = pre_norm_x
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=self.prenorm,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=self.prenorm,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            
            if self.prenorm:
                hidden_states, residual = hidden_states
        
        return hidden_states, residual

class AdaRMSNorm(AdaLayerNorm):
    def __init__(self, embedding_dim: int, num_embeddings: int, *args, drop_path=0, prenorm=True, **kwargs):
        super().__init__(embedding_dim, num_embeddings, *args, **kwargs)
        self.norm = FusedAddRMSNorm(embedding_dim, drop_path=drop_path, prenorm=prenorm, **kwargs)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, pre_norm_x=None, ) -> torch.Tensor:
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2)
        x, pre_norm_x = self.norm(x, pre_norm_x=pre_norm_x)
        x = x * (1 + scale) + shift
        return x, pre_norm_x

class AdaRMSNormContinuous(AdaLayerNormContinuous):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        *args, **kwargs,
    ):
        super().__init__(embedding_dim, conditioning_embedding_dim, *args, drop_path=0, prenorm=True, **kwargs)
        self.norm = FusedAddRMSNorm(embedding_dim, drop_path=drop_path, prenorm=prenorm, **kwargs)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor, pre_norm_x=None, ) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x, pre_norm_x = self.norm(x, pre_norm_x=pre_norm_x)
        x = x * (1 + scale)[:, None, :] + shift[:, None, :]
        return x, pre_norm_x

class AdaRMSNormZero(AdaLayerNormZero):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int, *args, rms_norm=True, drop_path=0, prenorm=True, **kwargs):
        super().__init__(embedding_dim, num_embeddings, *args, **kwargs) 
        self.norm = FusedAddRMSNorm(embedding_dim, rms_norm=rms_norm, drop_path=drop_path, prenorm=prenorm, **kwargs)
        # nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.LongTensor,
        hidden_dtype: Optional[torch.dtype] = None,
        pre_norm_x=None, 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        
        x, pre_norm_x = self.norm(x, pre_norm_x=pre_norm_x)
        x = x * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp, pre_norm_x

class AdaLNNormHalf(AdaRMSNormZero):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, *args, **kwargs):
        super().__init__(embedding_dim, *args, **kwargs) 
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.LongTensor,
        hidden_dtype: Optional[torch.dtype] = None,
        pre_norm_x=None, 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        
        x, pre_norm_x = self.norm(x, pre_norm_x=pre_norm_x)
        x = x * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, pre_norm_x

class AdaLayerNormHalf(AdaLayerNormZero):

    def __init__(self, embedding_dim: int, *args, **kwargs):
        super().__init__(embedding_dim, *args, **kwargs) 
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.LongTensor,
        hidden_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        
        x = self.norm(x)
        x = x * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa

class AdaRMSNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(
        self, 
        embedding_dim: int, 
        use_additional_conditions: bool = False,
        num_classes=None,
        class_dropout_prob=0,
    ):
        super().__init__()

        self.emb = PixArtAlphaEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions,
            num_classes=num_classes, class_dropout_prob=class_dropout_prob,
        )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
    
    def _init_weights(self):
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # No modulation happening here.
        embedded_timestep = self.emb(
            timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype,
        )
        return self.linear(self.silu(embedded_timestep)), embedded_timestep

from diffusers.models.embeddings import TimestepEmbedding, Timesteps
class PixArtAlphaEmbeddings(nn.Module):
    """
    For PixArt-Alpha.

    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(
        self, embedding_dim, size_emb_dim, 
        use_additional_conditions: bool = False, num_classes=None, class_dropout_prob=0,
    ):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions:
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
            self.aspect_ratio_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
        
        self.class_embedder = None
        if num_classes is not None:
            self.class_embedder = LabelEmbedding(num_classes, embedding_dim, dropout_prob=class_dropout_prob)

    def forward(self, timestep, resolution, aspect_ratio, batch_size, hidden_dtype, class_labels=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        if self.use_additional_conditions:
            resolution_emb = self.additional_condition_proj(resolution.flatten()).to(hidden_dtype)
            resolution_emb = self.resolution_embedder(resolution_emb).reshape(batch_size, -1)
            aspect_ratio_emb = self.additional_condition_proj(aspect_ratio.flatten()).to(hidden_dtype)
            aspect_ratio_emb = self.aspect_ratio_embedder(aspect_ratio_emb).reshape(batch_size, -1)
            conditioning = timesteps_emb + torch.cat([resolution_emb, aspect_ratio_emb], dim=1)
        else:
            conditioning = timesteps_emb

        if self.class_embedder is not None:
            class_labels = self.class_embedder(class_labels)
            conditioning = conditioning + class_labels

        return conditioning

class LabelEmbedding(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.

    Args:
        num_classes (`int`): The number of classes.
        hidden_size (`int`): The size of the vector embeddings.
        dropout_prob (`float`): The probability of dropping a label.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        #use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = torch.tensor(force_drop_ids == 1)
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels: torch.LongTensor, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (self.training and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings