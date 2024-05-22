# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.models.lora import LoRACompatibleLinear
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero #, RMSNorm
from diffusers.models.attention import FeedForward, BasicTransformerBlock

from copy import deepcopy

from mamba_ssm import Mamba
from mamba_attn_diff.models.adapter_attn4mamba import token_dropout

from .normalization import *

from einops import rearrange
from .freeu import apply_freeu

def _chunked_feed_forward(
    ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int, lora_scale: Optional[float] = None
):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    if lora_scale is None:
        ff_output = torch.cat(
            [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
            dim=chunk_dim,
        )
    else:
        # TOOD(Patrick): LoRA scale can be removed once PEFT refactor is complete
        ff_output = torch.cat(
            [ff(hid_slice, scale=lora_scale) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
            dim=chunk_dim,
        )

    return ff_output


@maybe_allow_in_graph
class GatedSelfAttentionDense(nn.Module):
    r"""
    A gated self-attention dense layer that combines visual features and object features.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        context_dim (`int`): The number of channels in the context.
        n_heads (`int`): The number of heads to use for attention.
        d_head (`int`): The number of channels in each head.
    """

    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))

        self.enabled = True

    def forward(self, x: torch.Tensor, objs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        n_visual = x.shape[1]
        objs = self.linear(objs)

        x = x + self.alpha_attn.tanh() * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, :n_visual, :]
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))

        return x

@maybe_allow_in_graph
class BasicMambaBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88, 
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        use_bidirectional_rnn=False,
        mamba_type='enc',
        nested_order=0,
        height=32, 
        width=32, 
        patch_size=2,
        interpolation_scale=1,
        is_skip=False,
        no_ff=False,
        ff_dim_mult=4, 
        use_conv1d=False,
        stage_index=0,
        in_channels=None,
        use_z_gate=True,
        use_reverse=False,
        extra_len=0,
        use_pad_token=False,
        rms=False,
        conv_dilation=1,
        drop_path=0,
        use_a4m_adapter=False,
        is_absorb=False,
        drop_rate=0,
        encoder_start_blk_id=1,
        num_layers=49,
        is_freeu=False, #skip_tune_val=-1,
        freeu_param=(0.3, 0.2, 1.1, 1.2),
        is_difffit=False,
        **kwargs,
    ):
        super().__init__()
        # self.skip_tune_val = skip_tune_val
        self.stage_index = stage_index
        self.extra_len = extra_len
        self.is_freeu = is_freeu
        self.freeu_param = freeu_param
        self.encoder_start_blk_id = encoder_start_blk_id
        self.num_layers = num_layers
        self.only_cross_attention = only_cross_attention

        self.use_bidirectional_rnn = use_bidirectional_rnn
        self.dim = dim
        self.in_channels = in_channels
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"
        self.use_ada_layer_norm_half = norm_type == "ada_norm_half"
        self.is_absorb = is_absorb
        self.drop_rate = drop_rate
        # assert (not is_absorb) or ( is_absorb and self.use_ada_layer_norm_zero)

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        elif positional_embeddings == "pos2d":
            from diffusers.models.embeddings import get_2d_sincos_pos_embed
            num_patches = (height // patch_size) * (width // patch_size)
            base_size = (height // patch_size)
            pos_embed = get_2d_sincos_pos_embed(
                dim, int(num_patches**0.5), 
                base_size=base_size, interpolation_scale=interpolation_scale,
            )
            self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)
        else:
            self.pos_embed = None

        self.rms = rms
        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaRMSNorm(dim, num_embeds_ada_norm, drop_path=drop_path,) if rms else AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaRMSNormZero(dim, num_embeds_ada_norm, drop_path=drop_path, rms_norm=False) if rms else AdaLayerNormZero(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_half:
            self.norm1 = AdaLNNormHalf(dim, num_embeds_ada_norm, drop_path=drop_path, rms_norm=False) if rms else AdaLayerNormHalf(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_continuous:
            self.norm1 = AdaRMSNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
                drop_path=drop_path,
            ) if rms else AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        elif norm_type is not None:
            self.norm1 = FusedAddRMSNorm(dim, drop_path=drop_path,) if rms else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        else:
            self.norm1 = None

        if mamba_type == 'enc':
            mamba_type = Mamba
        else:
            mamba_type = 'attn'
        
        mamba_config = dict(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dim, # Model dimension d_model
            d_state=mamba_d_state, #16,  # SSM state expansion factor
            d_conv=mamba_d_conv, #4,    # Local convolution width
            expand=mamba_expand, #2,    # Block expansion factor 
            nested_order=nested_order,
            pos_embed_h=height, 
            pos_embed_w=width, 
            patch_size=patch_size,
            interpolation_scale=interpolation_scale,
            use_conv1d=use_conv1d, use_bidirectional_rnn=self.use_bidirectional_rnn,
            stage_index=stage_index,
            in_channels=in_channels,
            use_z_gate=use_z_gate,
            use_reverse=use_reverse,
            extra_len=extra_len,
            use_pad_token=use_pad_token,
            conv_dilation=conv_dilation,
            no_ff=no_ff,
        )

        if mamba_type == 'attn':
            self.attn1 = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )
        else:
            self.attn1 = mamba_type(
                is_attn2=False,
                **mamba_config,
            ) 
        

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if self.use_ada_layer_norm:
                self.norm2 = AdaRMSNorm(dim, num_embeds_ada_norm, drop_path=drop_path,) if rms else AdaLayerNorm(dim, num_embeds_ada_norm)
            elif self.use_ada_layer_norm_continuous:
                self.norm2 = AdaRMSNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                    drop_path=drop_path,
                ) if rms else AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            else:
                self.norm2 = FusedAddRMSNorm(dim, drop_path=drop_path,) if rms else nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2 = mamba_type(
                is_attn2=True,
                **mamba_config,
            )
        else:
            if self.use_ada_layer_norm_single and self.is_absorb:
                self.norm2 = (
                    FusedAddRMSNorm(dim, drop_path=drop_path,) if rms else \
                        nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
                )
            else:
                self.norm2 = None
            
            self.attn2 = None

        self.no_ff = no_ff

        # 3. Feed-forward
        if self.use_ada_layer_norm_continuous:
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
            )
        elif not self.use_ada_layer_norm_single:
            if self.is_absorb:
                self.norm3 = FusedAddRMSNorm(dim, drop_path=drop_path,) if rms else nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        if not self.no_ff:
            self.ff = FeedForward(
                dim,
                mult=ff_dim_mult,
                dropout=dropout,
                activation_fn=activation_fn,
                final_dropout=final_dropout,
                inner_dim=ff_inner_dim,
                bias=ff_bias,
            )
        else:
            # if not (self.use_layer_norm or self.use_ada_layer_norm_half):
            if self.is_absorb:
                self.attn3 = mamba_type(**mamba_config)

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha.
        self.modulation_num = 6 if self.is_absorb else 3
        if self.use_ada_layer_norm_single:
            self.scale_shift_table = nn.Parameter(torch.randn(self.modulation_num, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        if self.dim <= 512:
            self.skip_linear = nn.Linear(2 * dim, dim) if is_skip else None
        else:
            if is_skip:
                self.skip_linear = nn.ModuleList([
                    nn.Linear(dim, dim),
                    nn.Linear(dim, dim),
                ])
            else:
                self.skip_linear = None
        
        self.is_difffit = is_difffit
        if self.is_difffit and stage_index <= self.num_layers + 1:
            self.gamma = nn.Parameter(torch.ones(dim))

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        pre_skip=None,
        pre_norm_x=None, 
        a4m_adapter=None,
        blk_id=0,
        is_fast = True,
    ) -> torch.FloatTensor:

        dtype = hidden_states.dtype

        # Notice that normalization is always applied before the real computation in the following blocks.
        rel_blk_id = 0
        def change_direction(x, pre_x, rel_blk_id=0, ): 
            # do not modify the sequence of pre_norm_x and pre_skip for smoother features.
            x, _, output_pre_norm_x = a4m_adapter(
                x, 
                pre_norm_x= pre_x, #if 'layerwise_cross' in a4m_adapter.sub_sequence_schedule else None, 
                block_idx=blk_id, 
                residual=None,
                pre_skip=None, #pre_skip,
                rel_blk_id=rel_blk_id,
            )
            # skip connect, x should be same sequence as x? no, alternative can make smooth feature !!!
            # if 'layerwise_cross' in a4m_adapter.sub_sequence_schedule:
            pre_x = output_pre_norm_x
            
            rel_blk_id += 1
            return x, pre_x, rel_blk_id

        # 0. Self-Attention
        batch_size = hidden_states.shape[0]
        if self.skip_linear is not None and pre_skip is not None:
            if self.rms:
                hidden_states = hidden_states + pre_norm_x
                hidden_states = hidden_states.to(dtype)
            
            if self.is_freeu:
                s1, s2, b1, b2 = self.freeu_param #0.3, 0.2, 1.1, 1.2
                hidden_states, pre_skip = apply_freeu(
                    resolution_idx=blk_id, 
                    hidden_states=hidden_states, 
                    res_hidden_states=pre_skip,
                    s1=s1, s2=s2, b1=b1, b2=b2,
                    encoder_start_blk_id=self.encoder_start_blk_id,
                    num_layers=self.num_layers,
                    extra_len=self.extra_len
                )

            if self.dim <= 512:
                hidden_states = self.skip_linear(
                    torch.cat([
                        hidden_states, 
                        pre_skip
                    ], dim=-1)
                )
            else:
                hidden_states = self.skip_linear[0](
                    hidden_states 
                ) + self.skip_linear[1]( 
                    pre_skip.to(dtype)
                )
            
            pre_norm_x = None
        
        if is_fast:
            if self.rms and (pre_norm_x is not None):
                hidden_states = hidden_states + pre_norm_x
                hidden_states = hidden_states.to(dtype)
            
            if a4m_adapter is not None:
                hidden_states, pre_norm_x, rel_blk_id = change_direction(
                    hidden_states, pre_x=None, rel_blk_id=rel_blk_id)
        
        if self.use_ada_layer_norm:
            if self.rms:
                norm_hidden_states, pre_norm_x = self.norm1(hidden_states, timestep, pre_norm_x=pre_norm_x)
            else:
                norm_hidden_states = self.norm1(hidden_states, timestep, )
            
        elif self.use_ada_layer_norm_zero:
            if self.rms:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, pre_norm_x = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype,
                    pre_norm_x=pre_norm_x,
                )
            else:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype,
                )
        elif self.use_ada_layer_norm_half:
            if self.rms:
                norm_hidden_states, gate_msa, pre_norm_x = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype,
                    pre_norm_x=pre_norm_x,
                )
            else:
                norm_hidden_states, gate_msa = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype,
                )

        elif self.use_layer_norm:
            if self.rms:
                norm_hidden_states, pre_norm_x = self.norm1(hidden_states, pre_norm_x=pre_norm_x, )
            else:
                norm_hidden_states = self.norm1(hidden_states,)

        elif self.use_ada_layer_norm_continuous:
            if self.rms:
                norm_hidden_states, pre_norm_x = self.norm1(
                    hidden_states, added_cond_kwargs["pooled_text_emb"], pre_norm_x=pre_norm_x)
            else:
                norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])

        elif self.use_ada_layer_norm_single:
            if self.is_absorb:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
            else:
                shift_msa, scale_msa, gate_msa = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 3, -1)
                ).chunk(3, dim=1)

            if self.rms:
                norm_hidden_states, pre_norm_x = self.norm1(hidden_states, pre_norm_x=pre_norm_x, )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            norm_hidden_states = hidden_states

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        if not is_fast:
            if a4m_adapter is not None:
                norm_hidden_states, pre_norm_x, rel_blk_id = change_direction(
                    norm_hidden_states, pre_norm_x, rel_blk_id=rel_blk_id)
        
        if self.drop_rate > 0:
            norm_hidden_states, \
            norm_hidden_states_drop_indices, \
            ori_norm_hidden_states = token_dropout(
                norm_hidden_states, 
                drop_rate=self.drop_rate,
                ori_x=None, 
                indices=None, 
                is_reverse=False, 
                training=self.training,
            )
        
        attn_output = self.attn1(
            norm_hidden_states, 
        )

        if self.drop_rate > 0:
            attn_output, _, _ = token_dropout(
                attn_output, 
                drop_rate=self.drop_rate,
                ori_x=ori_norm_hidden_states, 
                indices=norm_hidden_states_drop_indices, 
                is_reverse=True, 
                training=self.training,
            )


        if not self.no_ff:
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.use_ada_layer_norm_single:
                attn_output = gate_msa * attn_output

            hidden_states = attn_output + hidden_states
        else:
            if self.use_ada_layer_norm_half:
                attn_output = gate_msa.unsqueeze(1) * attn_output
                if self.rms:
                    hidden_states = attn_output
                else:
                    hidden_states = attn_output + hidden_states
            elif self.use_layer_norm:
                if self.rms:
                    hidden_states = attn_output
                else:
                    hidden_states = attn_output + hidden_states
            else:
                if self.use_ada_layer_norm_zero:
                    attn_output = gate_msa.unsqueeze(1) * attn_output
                elif self.use_ada_layer_norm_single:
                    attn_output = (1 + gate_msa) * attn_output
            
                if self.rms:
                    hidden_states = attn_output
                else:
                    hidden_states = attn_output + hidden_states
        
        if self.is_difffit:
            hidden_states = hidden_states * self.gamma
        
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.use_ada_layer_norm:
                if self.rms:
                    norm_hidden_states, pre_norm_x = self.norm2(hidden_states, timestep, pre_norm_x=pre_norm_x)
                else:
                    norm_hidden_states = self.norm2(hidden_states, timestep)

            elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                if self.rms:
                    norm_hidden_states, pre_norm_x = self.norm2(hidden_states, pre_norm_x=pre_norm_x)
                else:
                    norm_hidden_states = self.norm2(hidden_states)

            elif self.use_ada_layer_norm_single:
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.use_ada_layer_norm_continuous:
                if self.rms:
                    norm_hidden_states, pre_norm_x = self.norm2(
                        hidden_states, added_cond_kwargs["pooled_text_emb"], pre_norm_x=pre_norm_x)
                else:
                    norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])

            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        
        # 4. Feed-forward
        if self.is_absorb:
            if self.use_ada_layer_norm_continuous:
                if self.rms:
                    norm_hidden_states, pre_norm_x = self.norm3(attn_output, added_cond_kwargs["pooled_text_emb"], pre_norm_x=pre_norm_x)
                else:
                    norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif not self.use_ada_layer_norm_single:
                if self.rms:
                    norm_hidden_states, pre_norm_x = self.norm3(attn_output, pre_norm_x=pre_norm_x)
                else:
                    norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            if self.use_ada_layer_norm_single:
                if self.rms:
                    norm_hidden_states, pre_norm_x = self.norm2(attn_output, pre_norm_x=pre_norm_x)
                else:
                    norm_hidden_states = self.norm2(attn_output)

                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            if self.no_ff:
                attn3_output = self.attn3(
                    norm_hidden_states, 
                )
            else:
                if self._chunk_size is not None:
                    # "feed_forward_chunk_size" can be used to save memory
                    ff_output = _chunked_feed_forward(
                        self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size, lora_scale=lora_scale
                    )
                else:
                    ff_output = self.ff(norm_hidden_states, scale=lora_scale)
                
                attn3_output = ff_output

            if self.use_ada_layer_norm_zero:
                attn3_output = gate_mlp.unsqueeze(1) * attn3_output
            elif self.use_ada_layer_norm_single:
                attn3_output = gate_mlp * attn3_output
            
            hidden_states = attn3_output if self.rms else (attn3_output + hidden_states)

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        
        return hidden_states, pre_norm_x
