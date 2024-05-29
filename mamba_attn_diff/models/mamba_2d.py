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
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
from diffusers.configuration_utils import ConfigMixin, register_to_config
# from ..models.embeddings import ImagePositionalEmbeddings
# from ..utils import deprecate
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, is_torch_version
from diffusers.models.transformer_2d import Transformer2DModel
# from .lora import LoRACompatibleConv, LoRACompatibleLinear
# from .modeling_utils import ModelMixin
# from .normalization import AdaLayerNormSingle
from diffusers.models.embeddings import get_2d_sincos_pos_embed

from .attention import BasicMambaBlock

from .normalization import FusedAddRMSNorm, AdaRMSNorm, AdaRMSNormContinuous, AdaRMSNormZero, AdaRMSNormSingle

from functools import partial
from ..utils.init_weights import _init_weights_mamba, pos_embed_inteplot

@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


# class Transformer2DModel(ModelMixin, ConfigMixin):
class Mamba2DModel(Transformer2DModel):
    """
    A 2D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    @register_to_config
    def __init__(
        self, 
        num_attention_heads: int = 16, 
        attention_head_dim: int = 88,
        patch_size=None, 
        sample_size=None, 
        num_layers=1, 
        norm_type: str = "layer_norm",
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        use_bidirectional_rnn=False,
        mamba_type='enc',
        nested_order=0, 
        is_uconnect=True,
        no_ff=False,
        ff_dim_mult=4,
        in_channels=3,
        use_reverse=False,
        mlp_time_embed=False,
        is_extra_tokens=False,
        use_conv1d=False,
        use_pad_token=False,
        rms=False,
        conv_dilation=1,
        drop_path_rate=0., #0.1
        use_a4m_adapter=False,
        apply_adapter_last_linear=False,
        encoder_start_blk_id=0,
        num_2d_enc_dec_layers=0,
        kv_as_one_token_idx=-1,
        is_adapter_proj_in=False,
        use_final_conv=False,
        pad_token_schedules=['dec_split', 'lateral'],
        is_absorb=False,
        use_adapter_modules=True,
        nocat=False,
        sequence_schedule='reverse_concat', 
        sub_sequence_schedule=['simple'],
        pos_encoding_type='learnable',
        tkn_conv_dilated_rate=1,
        scan_pattern_len=3,
        is_align_exchange_q_kv=False,
        is_random_patterns=False,
        multi_times=1,
        pattern_type='base',
        drop_rate=0,
        pad_token_len=None,
        is_doubel_extra_tokens=False,
        num_patches=None,
        is_freeu=False,
        is_skip_tune=False,
        freeu_param=(0.3, 0.2, 1.1, 1.2),
        skip_tune_param = (0.82, 1.0),
        is_difffit=False,
        **kwargs
    ):
        super().__init__(
            num_attention_heads=num_attention_heads, 
            attention_head_dim=attention_head_dim,
            patch_size=patch_size, 
            sample_size=sample_size, 
            num_layers=num_layers, 
            norm_type=norm_type,
            in_channels=in_channels,
            **kwargs,
        )
        self.is_skip_tune = is_skip_tune
        self.patch_size = patch_size
        self.is_uconnect = is_uconnect
        inner_dim = num_attention_heads * attention_head_dim

        embed_dim = inner_dim
        self.inner_dim = inner_dim
        self.embed_dim = embed_dim
        self.extra_len = 0
        self.is_extra_tokens = is_extra_tokens
        self.pos_encoding_type = pos_encoding_type
        
        if num_patches is not None:
            if isinstance(num_patches, list) or isinstance(num_patches, tuple):
                max_num_patches = num_patches[0]
            else:
                max_num_patches = num_patches
        else:
            max_num_patches = (sample_size // patch_size) ** 2

        num_patches = max_num_patches if num_patches is None else num_patches

        num_embeds_ada_norm = self.config.num_embeds_ada_norm 
        if self.is_extra_tokens:
            extra_len = 0
            if num_embeds_ada_norm is not None:
                self.label_emb = nn.Embedding(num_embeds_ada_norm+1, embed_dim)
                extra_len += 1
            
            self.time_embed = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.SiLU(),
                nn.Linear(4 * embed_dim, embed_dim),
            ) if mlp_time_embed else nn.Identity()
            extra_len += 1

            if is_doubel_extra_tokens:
                extra_len += 2
            
            self.extra_len = extra_len
        
        if self.pos_encoding_type == 'learnable':
            self.additional_embed = nn.Parameter(torch.zeros(1, self.extra_len + max_num_patches, embed_dim))
        elif self.pos_encoding_type == 'rope':
            assert NotImplementedError
        elif self.pos_encoding_type == 'fourier': 
            pos_embed = get_2d_sincos_pos_embed(embed_dim, int(max_num_patches**0.5), )
            if self.extra_len > 0:
                pos_embed = np.concatenate([np.zeros([self.extra_len, embed_dim]), pos_embed], axis=0)

            self.register_buffer("additional_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)
        else:
            self.additional_embed = None

        self.pos_embed = UViTPatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            use_pos_embed=False,
        )
        # else:

        if norm_type == "ada_norm_single":
            self.use_additional_conditions = False #self.config.sample_size == 128
            # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
            # additional conditions until we find better name
            self.adaln_single = AdaRMSNormSingle(
                inner_dim, 
                use_additional_conditions=self.use_additional_conditions,
                num_classes=num_embeds_ada_norm,
                class_dropout_prob=0, # if dataloader has drop
            )
        
        if use_reverse:
            is_adapt_q_list = [False,] * num_layers
            is_learnable_q_list = [False,] * num_layers
            use_conv1d_list = [use_conv1d,] * num_layers
        else:
            is_adapt_q_list = [True,] * num_layers
            is_learnable_q_list = [False,] * num_layers
            use_conv1d_list = [use_conv1d,] * num_layers

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr

        self.rms = rms
        is_skip_list = [
            ( 
                self.is_uconnect and \
                ( d >= num_layers - (num_layers - encoder_start_blk_id)//2 )
            ) for d in range(num_layers)
        ]
        assert (num_layers - encoder_start_blk_id) % 2 == 0, 'TODO: this has not been correctly implemented yet.'
        # if (num_layers - encoder_start_blk_id) % 2 != 0:
        #     is_skip_list[ num_layers - (num_layers - encoder_start_blk_id)//2 ] = False

        collect_skip_list = [
            (
                self.is_uconnect and \
                (d >= encoder_start_blk_id) and \
                (d < encoder_start_blk_id + (num_layers - encoder_start_blk_id)//2)
            ) for d in range(num_layers)
        ]

        skip_top_val, skip_bottom_val = skip_tune_param
        skip_bottom_id, skip_top_id = num_layers, -1
        for d in range(num_layers):
            if is_skip_list[d]:
                skip_bottom_id = min(skip_bottom_id, d)
                skip_top_id = max(skip_top_id, d)
        
        self.skip_bottom_id = skip_bottom_id
        self.skip_top_id = skip_top_id
        self.skip_top_val = skip_top_val
        self.skip_bottom_val = skip_bottom_val

        # 3. Define transformers blocks
        transformer_block_param = dict()
        transformer_block_param.update(kwargs)
        transformer_block_param.update(
            dict(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                norm_type=norm_type, 
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                use_bidirectional_rnn=use_bidirectional_rnn,
                mamba_type=mamba_type,
                nested_order=nested_order,
                height=sample_size if sample_size is not None else 32, 
                width=sample_size if sample_size is not None else 32, 
                patch_size=patch_size if patch_size is not None else 2,
                interpolation_scale=max(1, sample_size // 64),
                no_ff=no_ff,
                ff_dim_mult=ff_dim_mult, 
                in_channels=None, 
                use_z_gate=True,
                use_reverse=use_reverse,
                extra_len=self.extra_len,
                use_pad_token=use_pad_token,
                rms=rms,
                conv_dilation=conv_dilation, 
                use_a4m_adapter=use_a4m_adapter,
                is_absorb=is_absorb,
                drop_rate=drop_rate,
                encoder_start_blk_id=encoder_start_blk_id,
                num_layers=num_layers,
                is_freeu=is_freeu,
                freeu_param=freeu_param,
                is_difffit=is_difffit,
            )
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicMambaBlock(
                    is_skip=is_skip_list[d],
                    stage_index=d,
                    is_adapt_q=( is_adapt_q_list[d]),
                    use_conv1d=( use_conv1d_list[d]),
                    drop_path=inter_dpr[d],
                    **transformer_block_param,
                )
                for d in range(num_layers)
            ]
        )

        self.no_ff = no_ff
        self.norm_last = None
        self.num_layers = num_layers

        self.encoder_start_blk_id = encoder_start_blk_id
        self.a4m_adapter = None
        if use_a4m_adapter:
            from .adapter_attn4mamba import AdapterAttnForMamba
            self.a4m_adapter = AdapterAttnForMamba(
                inner_dim, num_layers, 
                num_patches=num_patches,
                extra_len=self.extra_len,
                mamba_d_conv=mamba_d_conv,
                pad_token_len=pad_token_len ,  #-1
                sequence_schedule=sequence_schedule, 
                bias=True,
                use_pad_token=use_pad_token,
                apply_adapter_last_linear=apply_adapter_last_linear,
                num_2d_enc_dec_layers=num_2d_enc_dec_layers,
                kv_as_one_token_idx=kv_as_one_token_idx,
                is_skip_list = is_skip_list,
                is_proj_in=is_adapter_proj_in,
                mamba_type=mamba_type,
                pad_token_schedules=pad_token_schedules,
                use_adapter_modules=use_adapter_modules,
                nocat=nocat,
                sub_sequence_schedule=sub_sequence_schedule,
                is_absorb=is_absorb,
                encoder_start_blk_id=encoder_start_blk_id,
                tkn_conv_dilated_rate=tkn_conv_dilated_rate,
                scan_pattern_len=scan_pattern_len,
                is_align_exchange_q_kv=is_align_exchange_q_kv,
                is_random_patterns=is_random_patterns,
                multi_times=multi_times,
                pattern_type=pattern_type,
            )
        
        self.is_skip_list = is_skip_list
        self.collect_skip_list = collect_skip_list

        # 4. Define output layers
        self.scale_shift_table = None
        if self.is_input_continuous:
            pass
        elif self.is_input_vectorized:
            self.norm_out = FusedAddRMSNorm(
                inner_dim, drop_path=drop_path_rate, prenorm=False,) if rms else nn.LayerNorm(inner_dim)
        elif self.is_input_patches: #and norm_type != "ada_norm_single":
            self.proj_out = None
            self.norm_out = FusedAddRMSNorm(
                inner_dim, drop_path=drop_path_rate, prenorm=False,
            ) if rms else nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_2 = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)
            if self.is_extra_tokens:
                self.proj_out_1 = nn.Linear(inner_dim, 2 * inner_dim)
            elif norm_type == "ada_norm_single":
                self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)

        self.use_final_conv = use_final_conv
        if self.use_final_conv:
            self.final_layer = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)

        self._init_weights()
    
    def get_skiptune_weight(self, skip_bottom_val, skip_top_val):
        num_layers = self.num_layers
        skip_bottom_id, skip_top_id = self.skip_bottom_id, self.skip_top_id
        skip_tune_val_list = [
            (
                (
                    (d - skip_bottom_id)*1./ (skip_top_id - skip_bottom_id) * (skip_top_val - skip_bottom_val) + skip_bottom_val
                ) if self.is_skip_tune and self.is_skip_list[d] else None
            ) for d in range(num_layers) 
        ]
        return skip_tune_val_list

        
    def _init_weights(self, initializer_cfg=None):
        self.apply(
            partial(
                _init_weights_mamba,
                n_layer=self.num_layers,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'additional_embed'} if self.is_extra_tokens else {}

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        y: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        use_pos_inteplot=False,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        dtype = hidden_states.dtype
        input_timestep = timestep
        if class_labels is None and self.config.num_embeds_ada_norm is not None:
            class_labels = y if y is not None else torch.zeros_like(timestep).long()+self.config.num_embeds_ada_norm

        # print(timestep, class_labels)
        # assert False

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        def create_custom_forward(module, return_dict=None):
            def custom_forward(*inputs):
                if return_dict is not None:
                    return module(*inputs, return_dict=return_dict)
                else:
                    return module(*inputs)

            return custom_forward

        # 1. Input
        if self.is_input_continuous:
            batch, _, height, width = hidden_states.shape
            residual = hidden_states

            hidden_states = self.norm(hidden_states)
            if not self.use_linear_projection:
                hidden_states = (
                    self.proj_in(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_in(hidden_states)
                )
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                hidden_states = (
                    self.proj_in(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_in(hidden_states)
                )

        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
            hidden_states = self.pos_embed(hidden_states)

            if self.adaln_single is not None:
                if self.use_additional_conditions and added_cond_kwargs is None:
                    raise ValueError(
                        "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                    )
                elif added_cond_kwargs is None:
                    added_cond_kwargs = dict(
                        resolution=None, 
                        aspect_ratio=None,
                        class_labels=class_labels,
                    )

                batch_size = hidden_states.shape[0]
                timestep, embedded_timestep = self.adaln_single(
                    input_timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype,
                )

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        if self.is_extra_tokens: #and self.adaln_single is None:
            embedded_timestep = self.time_embed(timestep_embedding(input_timestep, self.embed_dim))

        if self.extra_len == 1:
            hidden_states = torch.cat([
                embedded_timestep[:,None,:],
                hidden_states, 
            ], dim=-2)
        elif self.extra_len == 2:
            class_embed = self.label_emb(class_labels)
            hidden_states = torch.cat([
                embedded_timestep[:,None,:], 
                class_embed[:,None,:], 
                hidden_states, 
            ], dim=-2)
        elif self.extra_len == 4:
            class_embed = self.label_emb(class_labels)
            hidden_states = torch.cat([
                embedded_timestep[:,None,:], 
                class_embed[:,None,:], 
                class_embed[:,None,:], 
                embedded_timestep[:,None,:], 
                hidden_states, 
            ], dim=-2)
        
        if self.additional_embed is not None:
            additional_embed = self.additional_embed
            if use_pos_inteplot or (height*width != additional_embed.shape[-2] - self.extra_len):
                additional_embed = pos_embed_inteplot(
                    cur_pos_embed=None, 
                    pretrained_pos_embed=additional_embed, 
                    extra_len=self.extra_len, 
                    cur_size=(height, width),
                )

            hidden_states = hidden_states + additional_embed

        skips = []
        skips_id = []
        pre_norm_x = None

        hidden_states = hidden_states.to(dtype)

        a4m_adapter = self.a4m_adapter
        if a4m_adapter is not None:
            hidden_states, pre_norm_x = a4m_adapter.prepare_sequence(hidden_states, height=height)
        
        skip_tune_val_list = None
        if self.is_skip_tune:

            t_continuous = torch.ones_like(input_timestep).reshape(-1, 1, 1) / 1000
            
            # # increase rho
            # skip_top_val = self.skip_top_val + ( self.skip_bottom_val - self.skip_top_val ) / 1 * t_continuous # 0.82
            # decrease rho
            skip_top_val = self.skip_bottom_val + ( self.skip_top_val - self.skip_bottom_val ) / 1 * t_continuous

            skip_bottom_val = self.skip_bottom_val # 1

            skip_tune_val_list = self.get_skiptune_weight(skip_bottom_val=skip_bottom_val, skip_top_val=skip_top_val)

        for blk_id, block in enumerate(self.transformer_blocks):
            pre_skip = None 
            if self.is_skip_list[blk_id]: 
                pre_skip = skips.pop()
                pre_skip_id = skips_id.pop()

                if (skip_tune_val_list is not None) and (skip_tune_val_list[blk_id] is not None):
                    
                    skip_tune_val = skip_tune_val_list[blk_id]
                    # print(pre_skip.shape, skip_tune_val, input_timestep, input_timestep.shape)
                    pre_skip = pre_skip * skip_tune_val
                    

            if self.training and self.gradient_checkpointing:
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, pre_norm_x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    added_cond_kwargs,
                    pre_skip,
                    pre_norm_x,
                    a4m_adapter,
                    blk_id,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, pre_norm_x = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    added_cond_kwargs=added_cond_kwargs,
                    pre_skip=pre_skip,
                    pre_norm_x=pre_norm_x,
                    a4m_adapter=a4m_adapter,
                    blk_id=blk_id,
                )

            if self.collect_skip_list[blk_id]:
                skips.append( (hidden_states + pre_norm_x) if self.rms else hidden_states )
                skips_id.append(blk_id)

        hidden_states = hidden_states.to(dtype)
        if a4m_adapter is not None:
            hidden_states, pre_norm_x = a4m_adapter.finalize_sequence(
                hidden_states, pre_norm_x=pre_norm_x, height=height)
        elif a4m_adapter is None:
            # if self.norm_last is not None:
            #     hidden_states = self.norm_last(hidden_states, pre_norm_x=pre_norm_x)[0]
            # else:
            #     hidden_states = (hidden_states + pre_norm_x) if self.rms else hidden_states
            hidden_states = hidden_states[:, self.extra_len:, :]
            pre_norm_x = pre_norm_x[:, self.extra_len:, :] if pre_norm_x is not None else None

        # 3. Output
        if self.is_input_continuous:

            if not self.use_linear_projection:
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                hidden_states = (
                    self.proj_out(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_out(hidden_states)
                )
            else:
                hidden_states = (
                    self.proj_out(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_out(hidden_states)
                )
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

            output = hidden_states + residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states, pre_norm_x=pre_norm_x)[0] if self.rms else self.norm_out(hidden_states)

            logits = self.out(hidden_states)
            # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0))
            output = F.log_softmax(logits.double(), dim=1).float()

        if self.is_input_patches:

            if self.is_extra_tokens:
                conditioning = embedded_timestep
                if self.extra_len >= 2:
                    conditioning = conditioning + class_embed
                
                conditioning = conditioning.to(dtype)
                shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            elif self.config.norm_type == "ada_norm_single":
                shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
            else:
                conditioning = self.transformer_blocks[0].norm1.emb(
                    timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            
            hidden_states = self.norm_out(hidden_states, pre_norm_x=pre_norm_x)[0] if self.rms else self.norm_out(hidden_states)

            if self.is_extra_tokens:
                hidden_states = hidden_states * (1 + scale[:, None]) + shift[:, None]
                hidden_states = hidden_states.to(dtype)
                hidden_states = self.proj_out_2(hidden_states)
            elif self.config.norm_type == "ada_norm_single":
                # Modulation
                hidden_states = hidden_states * (1 + scale) + shift
                hidden_states = self.proj_out_2(hidden_states)
                hidden_states = hidden_states.squeeze(1)

            # unpatchify
            if self.adaln_single is None:
                height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )
            if self.use_final_conv:
                output = self.final_layer(output)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class UViTPatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        use_pos_embed=True,
    ):
        super().__init__()

        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.use_pos_embed = use_pos_embed
        if self.use_pos_embed:
            pos_embed = get_2d_sincos_pos_embed(embed_dim, int(num_patches**0.5))
            self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

    def forward(self, latent):
        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)
        if self.use_pos_embed:
            return latent + self.pos_embed
        else:
            return latent

class NTKLlamaRotaryEmbedding(nn.Module):
    # def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
    def __init__(
        self, 
        dim, 
        max_position_embeddings=16384, base=10000, device=None, scaling_factor=1.0, s=1,
    ): # a = 8 #Alpha value
        super().__init__()

        base = base * max(1, s) ** (dim / (dim-2))

        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

    @property
    def sin_cached(self):
        logger.warning_once(
            "The sin_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self._sin_cached

    @property
    def cos_cached(self):
        logger.warning_once(
            "The cos_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self._cos_cached

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        if seq_len is not None:
            logger.warning_once("The `seq_len` argument is deprecated and unused. It will be removed in v4.39.")

        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)