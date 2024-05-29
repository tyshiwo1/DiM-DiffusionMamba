from copy import deepcopy
from mamba_ssm import Mamba
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
import torch.utils.checkpoint

from einops import rearrange, repeat

from diffusers.models.embeddings import SinusoidalPositionalEmbedding, get_2d_sincos_pos_embed

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

# ft baseline
def _init_base_indices(H, W, pad_len, eps=1e-3, dtype=torch.float32, multi_times=2):
    elem_init_position_in_x_seq = dict()
    x_seq_position_with_img_arrange = []
    pad_indices = []
    cnt = 0 
    for ii in range(0, 1):
        for jj in range(0, 1):
            for i in range(0, H + pad_len):
                for j in range(0, W + pad_len):
                    if (i >= H) or (j >= W):
                        pad_indices.append(cnt)
                    
                    elem_init_position_in_x_seq[ (ii, jj, i, j) ] = cnt
                    cnt += 1

    for i in range(0, H ):
        for j in range(0, W ):
            for ii in range(0, 1):
                for jj in range(0, 1):
                    if (ii, jj, i, j) in elem_init_position_in_x_seq:
                        x_seq_position_with_img_arrange.append(elem_init_position_in_x_seq[ (ii, jj, i, j) ])

    x_seq_position_with_img_arrange = torch.tensor(x_seq_position_with_img_arrange, dtype=torch.long)
    pad_indices = torch.tensor(pad_indices, dtype=torch.long)
    return elem_init_position_in_x_seq, x_seq_position_with_img_arrange, pad_indices

# the sequence of seq_idx should correspond to the formation of elem_init_position_in_x_seq
def get_multi_times_seq_different_order(
    list1=range(0, 1), list2=range(0, 1), 
    listA=range(0, 1), listB=range(0, 1), 
    elem_init_position_in_x_seq={}, 
    ij_temp=0, 
    **kwargs,
):
    '''
        if vertical first ((ij_temp + 2) % 2 == 1), exchange (i,j), (ii, jj)
    '''
    seq_idx = []
    for ii in listA:
        for jj in listB:
            for i in list1:
                for j in list2:
                    point_position = (ii, jj, i, j) if (ij_temp + 2) % 2 == 0 else (jj, ii, j, i)
                    if point_position in elem_init_position_in_x_seq:
                        seq_idx.append( elem_init_position_in_x_seq[ point_position ] )

    # print(listA, listB, list1, list2,  )
    assert len(seq_idx) == len(elem_init_position_in_x_seq), "seq_idx should have the same number of tokens as elem_init_position_in_x_seq {} {} {}".format(len(seq_idx), len(elem_init_position_in_x_seq), ij_temp)
    
    return seq_idx


def aug_pad_tokens(pads, pad_len, seq_pad_inds):
    C = pads.shape[1]
    pads = rearrange(pads, 'l c n -> l n c').reshape(-1, 1, pad_len, C).expand(
        pads.shape[0], len(seq_pad_inds)//pad_len, pad_len, C
    )
    pads = pads.reshape(-1, len(seq_pad_inds), C)
    return pads

def adaptive_fill_pad_into_sequence(
    x, x_seq_position_with_img_arrange, pad_indices, pad_token, pad_len, 
    width,
):
    x_new = x.new_zeros(x.shape[0], len(x_seq_position_with_img_arrange)+len(pad_indices), *x.shape[2:])
    imgsize_x = rearrange(x, 'b (h w) c -> b c h w', w=width)
    
    x_new[:, x_seq_position_with_img_arrange, :] = x 
    # x_seq_position_with_img_arrange must correspond to continous elements in x's space

    x_new[:, pad_indices, :] = aug_pad_tokens(pad_token, pad_len, pad_indices).to(x.dtype)
    return x_new

    
def relative_scan_pattern_transformation(x, last_pattern, cur_pattern, height=None, dim=-2):
    '''
        def horizon_forward(x, height=None): # 0
            return x
        
        def vertical_forward(x, height=None): # 1
            x = rearrange(x, 'b (h w) c -> b (w h) c', h=height)
            return x
        
        def horizon_backward(x, height=None): # -2
            return x.flip(-2)
        
        def vertical_backward(x, height=None): # -1
            x = rearrange(x, 'b (w h) c -> b (h w) c', h=height).flip(-2)
            return x
    '''
    if x.shape[dim] == 0:
        return x

    if last_pattern < 0 and cur_pattern >= 0 : # ('backward' in last_pattern) and ('backward' not in cur_pattern):
        x = x.flip(dim)
    
    if (last_pattern + cur_pattern + 4) % 2 == 1 : #if cur_pattern.split('_')[0] != last_pattern.split('_')[0]:
        if (cur_pattern + 4) % 2 == 1 : #'vertical' in cur_pattern:
            if dim == 0:
                x = rearrange(x, '(h w) -> (w h)', h=height)
            else:
                x = rearrange(x, 'b (h w) c -> b (w h) c', h=height)
        elif (cur_pattern + 4) % 2 == 0 : #'horizontal' in cur_pattern:
            if dim == 0:
                x = rearrange(x, '(w h) -> (h w)', h=height)
            else:
                x = rearrange(x, 'b (w h) c -> b (h w) c', h=height)
            
    if cur_pattern < 0 and last_pattern >= 0 : # ('backward' in cur_pattern) and ('backward' not in last_pattern):
        x = x.flip(dim)
    
    return x

def get_seq_different_order(list1, list2, elem_init_position_in_x_seq, ij_temp=0):
    pre_seq_idx, post_seq_idx = [], []
    all_pad_token_tmp = 0
    for i in list1:
        for j in list2:
            ij = (i, j) if (ij_temp + 2) % 2 == 0 else (j, i)
            post_seq_idx.append( elem_init_position_in_x_seq[ij] )

    return pre_seq_idx, post_seq_idx

def get_relative_permutation(last_abso_permu, cur_abso_permu):
    return [ cur_abso_permu.index(i) for i in last_abso_permu ]
    # return np.argsort(np.argsort(cur_abso_permu))[np.argsort(last_abso_permu)].tolist()

class AdapterAttnForMamba(nn.Module):
    def __init__(
        self, dim, num_layers, num_patches=256, extra_len=0, 
        mamba_d_conv=3, pad_token_len=None, sequence_schedule='dilated', bias=True,
        conv_kernel_size=3, conv_bias=True, use_pad_token=False, 
        pad_token_schedules=['dec_split', 'lateral'],
        apply_adapter_last_linear=False, num_2d_enc_dec_layers=0,
        kv_as_one_token_idx=-1, is_skip_list=[], skip_interval=2,
        is_proj_in=True, mamba_type='enc',
        use_adapter_modules=True,
        nocat=False,
        sub_sequence_schedule='simple',
        is_absorb=False,
        encoder_start_blk_id=0,
        tkn_conv_dilated_rate=1,
        scan_pattern_len=3,
        is_align_exchange_q_kv=False,
        is_random_patterns=False,
        multi_times=1,
        pattern_type='base',
        **kwargs,
    ):
        super().__init__()
        pad_token_len = pad_token_len if pad_token_len is not None else mamba_d_conv
        self.is_absorb = is_absorb
        self.apply_adapter_last_linear = apply_adapter_last_linear
        
        # num_layers = num_layers if self.apply_adapter_last_linear else 1
        self.num_adapters = ( min(kv_as_one_token_idx, num_layers) if kv_as_one_token_idx>0 else num_layers) \
            if not self.apply_adapter_last_linear else 1
        self.num_layers = num_layers
        self.num_2d_enc_dec_layers = num_2d_enc_dec_layers
        self.kv_as_one_token_idx = kv_as_one_token_idx
        self.is_proj_in = is_proj_in
        self.mamba_type = mamba_type
        self.use_adapter_modules = use_adapter_modules
        self.nocat = nocat
        self.sequence_schedule = sequence_schedule
        self.scan_pattern_len = scan_pattern_len
        self.is_align_exchange_q_kv = is_align_exchange_q_kv
        self.is_random_patterns = is_random_patterns
        assert (not is_random_patterns)
        self.encoder_start_blk_id = encoder_start_blk_id
        self.multi_times = multi_times
        self.pattern_type = pattern_type
        register_layers_list = [True, ] if self.apply_adapter_last_linear else is_skip_list

        conv_dim = dim
        conv_kwargs = dict(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            kernel_size=conv_kernel_size,
        )

        if self.use_adapter_modules:
            self.first_smoother = nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                groups=dim,
                kernel_size=3,
                padding=tkn_conv_dilated_rate, #(3-1)//2,
                bias=conv_bias,
                dilation = tkn_conv_dilated_rate,
            )
            self.last_smoother = nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                groups=dim,
                kernel_size=3,
                padding=tkn_conv_dilated_rate, #(3-1)//2,
                bias=conv_bias,
                dilation = tkn_conv_dilated_rate,
            )

        self.extra_len = extra_len

        
        self.sub_sequence_schedule = sub_sequence_schedule
        self.input_seq_len = None

        self.use_pad_token = use_pad_token
        self.pad_token_schedules = pad_token_schedules
        
        if self.use_pad_token:
            self.pad_token_len = pad_token_len
            if ('dec_split' in self.pad_token_schedules) and self.extra_len > 0:
                self.pad_token_dec_split = nn.Parameter(torch.randn(1, self.pad_token_len, dim)*0.02, requires_grad=True)
            else:
                self.pad_token_dec_split = torch.zeros(1, 0, dim)
            
            if any(x in self.pad_token_schedules for x in ['lateral', 'embedin_lateral', 'rho_pad']):
                self.pad_token = nn.Parameter(torch.randn(1, dim, self.pad_token_len)*0.02, requires_grad=True)
        else:
            self.pad_token_len = 0
            self.pad_token_dec_split = torch.zeros(1, 0, dim)
        
        self.flip_patterns = None

        seq_lateral_inds = None

        init_pattern_indices_func_type = _init_base_indices
        

        assert is_absorb or (self.num_layers % 2 == 1), 'num_layers should be odd'

        pad_token_len = self.pad_token_len
        
            
        self._init_multiscale_sequence_idx_variables(
            num_patches, pad_token_len, 
            multi_times, pattern_type, 
            encoder_start_blk_id, scan_pattern_len, num_layers,
            sub_sequence_schedule,
        )
                
        
    def _init_multiscale_sequence_idx_variables(
        self, num_patches, pad_token_len, 
        multi_times, pattern_type, 
        encoder_start_blk_id, scan_pattern_len, num_layers,
        sub_sequence_schedule,
    ):
        self.num_patches_list = [] 
        self.seq_idx_sets = dict()
        input_num_patches_list = num_patches if isinstance(num_patches, list) or isinstance(num_patches, tuple) else [num_patches, ]
        for num_patches in input_num_patches_list:
            W = int(num_patches**0.5)
            H = num_patches // W
            assert num_patches // W * W == num_patches, "num_patches should be a square number {}, {}, {}".format(input_num_patches_list, num_patches, W)
            self.seq_idx_sets[ (H, W) ] = SequenceIndexSet(
                pad_token_len, multi_times, pattern_type,
                encoder_start_blk_id, scan_pattern_len, num_layers,
                sub_sequence_schedule,
            )
            self.num_patches_list.append( (H, W) )
            # print(f'num_patches {num_patches} H {H} W {W}')

        for num_patches in self.num_patches_list: 
            H, W = num_patches
            self.seq_idx_sets[ num_patches ]._init_sequence_idx_variables(H, W)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"additional_embed", "pad_token"}
    
    def forward(self, *args, **kwargs):
        if self.scan_pattern_len <= 0:
            return self.forward_pure(*args, **kwargs)
        elif self.pattern_type == 'base':
            return self.forward_fast(*args, **kwargs)
        else:
            return self.forward_ptr(*args, **kwargs)
    
    def forward_pure(self, x, pre_norm_x=None, block_idx=0, residual=None, pre_skip=None, rel_blk_id=0):
        return x, pre_skip, pre_norm_x
    
    def forward_fast(self, x, pre_norm_x=None, block_idx=0, residual=None, pre_skip=None, rel_blk_id=0):
        block_idx = block_idx*2 + rel_blk_id if self.is_absorb else block_idx
        scan_pattern = self.scan_pattern_chain[block_idx]
        last_pattern, cur_pattern = scan_pattern

        bs, L = x.shape[:2]
        
        def sequence_scan_change(seq, dim=-2):
            extra_token, kv, dec_split_pad, q = self.split_template(seq, dim=dim)

            if self.is_align_exchange_q_kv: # must false to stabilize huge model training
                kv_rev, q = q, kv
            else:
                q, kv_rev = q, kv

            kv_rev = relative_scan_pattern_transformation(
                kv_rev, last_pattern, cur_pattern, 
                height=self.height + self.pad_token_len, dim=dim)
            q = relative_scan_pattern_transformation(
                q, last_pattern, cur_pattern, 
                height=self.height + self.pad_token_len, dim=dim)

            x_concat_list = self.concat_template(
                q, extra_token=extra_token, 
                dec_split_pad=dec_split_pad, 
                pop_last=False, 
                aug_seq=kv_rev, 
                is_expand_split=False,
            )
            seq = torch.cat(x_concat_list, dim=dim) if len(x_concat_list) > 1 else x_concat_list[-1]
            return seq, q
        
        x, q = sequence_scan_change(x, dim=-2)

        if pre_norm_x is not None:
            pre_norm_x, _ = sequence_scan_change(pre_norm_x, dim=-2)
        
        if pre_skip is not None:
            pre_skip, _ = sequence_scan_change(pre_skip, dim=-2)
        
        if residual is not None:
            residual, _ = sequence_scan_change(residual, dim=-2)
        
        if residual is not None:
            x = residual + x
        
        if self.is_align_exchange_q_kv:
            self.now_pre_seq_idx, self.now_post_seq_idx = self.now_post_seq_idx, self.now_pre_seq_idx
        
        self.input_seq_len = q.shape[-2]

        return x, pre_skip, pre_norm_x
    
    def forward_ptr(self, x, pre_norm_x=None, block_idx=0, residual=None, pre_skip=None, rel_blk_id=0):
        block_idx = block_idx*2 + rel_blk_id if self.is_absorb else block_idx

        bs, L = x.shape[:2]

        x_feats = x
        
        x_ptr = torch.arange(L, dtype=torch.long, device=x.device)
        
        extra_token, kv, dec_split_pad, q = self.split_template(x_ptr, dim=0)

        if False:
            kv_rev, q = q, kv
            self.now_pre_seq_idx, self.now_post_seq_idx = self.now_post_seq_idx, self.now_pre_seq_idx
        else:
            q, kv_rev = q, kv
        
        self.input_seq_len = q.shape[0]
        
        if 'reverse_single' in self.sub_sequence_schedule:
            self.now_pre_seq_idx = self.acc_flip(self.now_pre_seq_idx, 0, block_idx=block_idx)
            self.now_post_seq_idx = self.acc_flip(self.now_post_seq_idx, 0, block_idx=block_idx) 
            kv_rev, q = self.acc_flip(kv_rev, 0, block_idx=block_idx), self.acc_flip(q, 0, block_idx=block_idx)

        x_concat_list = self.concat_template(
            q, extra_token=extra_token, dec_split_pad=dec_split_pad, pop_last=False, aug_seq=kv_rev, is_expand_split=False,)
        x_ptr = torch.cat(x_concat_list, dim=0) if len(x_concat_list) > 1 else x_concat_list[-1]
        
        x = acc_index_select(x, dim=1, index=x_ptr)

        if pre_norm_x is not None:
            pre_norm_x = acc_index_select(pre_norm_x, dim=1, index=x_ptr)
        
        if pre_skip is not None:
            pre_skip = acc_index_select(pre_skip, dim=1, index=x_ptr)
        
        if residual is not None:
            residual = acc_index_select(residual, dim=1, index=x_ptr)
        
        if residual is not None:
            x = residual + x 

        return x, pre_skip, pre_norm_x

    def prepare_sequence(self, x, height=None):
        bs, L = x.shape[:2]
        extra_token = x[:, :self.extra_len, :]
        x = x[:, self.extra_len:, :]
        
        self.height = height
        self.width = x.shape[-2] // height

        img_size = (height, self.width)
        if isinstance(img_size[0], torch.Tensor):
            img_size = (height.item(), self.width.item() )

        x_seq_position_with_img_arrange = self.seq_idx_sets[img_size].x_seq_position_with_img_arrange 
        pad_indices = self.seq_idx_sets[img_size].pad_indices 
        pre_seq_idx = self.seq_idx_sets[img_size].pre_seq_idx 
        post_seq_idx = self.seq_idx_sets[img_size].post_seq_idx
        self.elem_init_position_in_x_seq = self.seq_idx_sets[img_size].elem_init_position_in_x_seq
        self.flip_patterns =  self.seq_idx_sets[img_size].flip_patterns
        self.scan_pattern_chain = self.seq_idx_sets[img_size].scan_pattern_chain
        self.absolute_patterns = self.seq_idx_sets[img_size].absolute_patterns

        if self.use_adapter_modules: 
            x = apply_conv2d_bld(x, self.first_smoother, height)

        pre_norm_x = None

        if self.use_pad_token:
            if 'lateral' in self.pad_token_schedules:
                x, width, L = lateral_token_pad(self.pad_token, x, self.pad_token_len, height)
            elif 'embedin_lateral' in self.pad_token_schedules:
                x = embedin_lateral_add(self.pad_token, x, self.pad_token_len, height, self.seq_lateral_inds)

        self.input_seq_len = len(post_seq_idx)
        self.now_pre_seq_idx = pre_seq_idx
        self.now_post_seq_idx = post_seq_idx
        self.now_x_seq_position_with_img_arrange = x_seq_position_with_img_arrange
        self.now_pad_indices = pad_indices

        if len(self.now_pre_seq_idx) == 0:
            x_pre = x[:, 0:0, :]
            x_post = x
            self.input_seq_len = x.shape[-2] 
        elif len(self.now_post_seq_idx) == 0:
            x_pre = x
            x_post = x[:, 0:0, :]
            self.input_seq_len = x.shape[-2]
        else:
            x_pre = x[:, self.now_pre_seq_idx, :]
            x_post = x[:, self.now_post_seq_idx, :]

        if self.use_pad_token:
            if 'rho_pad' in self.pad_token_schedules:
                x_post = adaptive_fill_pad_into_sequence(
                    x, #x_post, 
                    x_seq_position_with_img_arrange=x_seq_position_with_img_arrange, 
                    pad_indices=pad_indices, 
                    pad_token=self.pad_token, 
                    pad_len=self.pad_token_len,
                    width=self.width,
                )
                self.input_seq_len = x_post.shape[-2]
        
        x_concat_list = self.concat_template(
            x_post, extra_token=extra_token, dec_split_pad=self.pad_token_dec_split,
            pop_last=False, aug_seq=x_pre,
        )
        x = torch.cat(x_concat_list, dim=-2) if len(x_concat_list) > 1 else x_concat_list[-1]

        return x, pre_norm_x
    
    def finalize_sequence(self, x, pre_norm_x=None, height=None):
        if x is None:
            return x, pre_norm_x
        
        dtype = x.dtype

        x = x if pre_norm_x is None else (pre_norm_x + x)
        pre_norm_x = None
        
        extra_token, x_pre, dec_split_pad, x_post = self.split_template(x)
        
        if 'rho_pad' in self.pad_token_schedules:

            if (self.flip_patterns is None) and ('reverse_single' in self.sub_sequence_schedule):
                x_post = self.acc_flip(x_post, dim=-2)

            # This must be work when the last pattern is 0 => x_seq_position_with_img_arrange is in same as the inital state
            x = x_post[:, self.now_x_seq_position_with_img_arrange, :]
            
        elif len(self.now_pre_seq_idx) == 0 or len(self.now_post_seq_idx) == 0:
            assert self.scan_pattern_chain[-1][1] == 0, 'last pattern must be the forward'
            x = x_post if len(self.now_pre_seq_idx) == 0 else x_pre
        else:
            # x_restored_shape = self.x_restored_shape
            x_restored_shape = self.self.seq_idx_sets[img_size].x_restored_shape
            x = x_post.new_zeros(
                x_post.shape[0], 
                x_restored_shape, 
                x_post.shape[-1]
            )
            if len(self.now_pre_seq_idx) > 0:
                x[:, self.now_pre_seq_idx, :] += x_pre
            
            if len(self.now_post_seq_idx) > 0:
                x[:, self.now_post_seq_idx, :] += x_post

        if self.use_pad_token:
            if 'lateral' in self.pad_token_schedules:
                # for now, it must be 0 pattern !!!
                x, width, paded_feats = lateral_token_unpad(x, self.pad_token_len, height)
        
        x = x.to(dtype)
        if self.use_adapter_modules:
            # print('2', x.device, x.shape, 'x.device', self.last_smoother.weight.device, self.last_smoother.weight.shape, 'self.last_smoother.weight.device')
            x = apply_conv2d_bld(x, self.last_smoother, height)

        return x, pre_norm_x

    def concat_template(self, x, extra_token=None, dec_split_pad=None, pop_last=False, aug_seq=None, is_expand_split=True,):
        x_concat_list = []
            
        if self.extra_len > 0 and extra_token.numel() > 0:
            x_concat_list = [extra_token, ]
        
        if self.mamba_type == 'enc' or aug_seq is None: 
            x_concat_list.append( self.acc_flip(x, -2) if aug_seq is None else aug_seq )
        
        if 'dec_split' in self.pad_token_schedules and dec_split_pad.numel() > 0:
            x_concat_list.append(
                dec_split_pad.expand(x.shape[0], self.pad_token_len, *x.shape[2:]) if is_expand_split else \
                    dec_split_pad
            )
        
        if not pop_last:
            x_concat_list.append(x)
        
        return x_concat_list
    
    def split_template(self, x, dim=-2):
        extra_token, kv, dec_split_pad, q = torch.split(
            x, [
                self.extra_len,
                x.shape[dim] - self.input_seq_len - self.extra_len - self.pad_token_dec_split.shape[-2], 
                self.pad_token_dec_split.shape[-2], 
                self.input_seq_len,
            ], dim=dim
        )
        return extra_token, kv, dec_split_pad, q

    def acc_flip(self, x, dim=0, block_idx=None):
        if x.numel() == 0:
            return x
        elif (block_idx is None) or (self.flip_patterns is None): ####!!!!!!!!!!!
            return x.flip(dim)
        elif len(x.shape)>=3 and x.shape[-2] > 0:
            return x[..., self.flip_patterns[block_idx], :]
        elif len(x.shape)==1:
            return x[self.flip_patterns[block_idx]]
        else:
            assert False


def add_distant_grad_skip_connection(x, width):
    return x
    
def apply_conv2d_bld(x, conv_layer, height):
    x = rearrange(x, 'b (h w) c -> b c h w', h=height).contiguous()
    x = conv_layer(x)
    x = rearrange(x, 'b c h w -> b (h w) c')
    return x

def embedin_lateral_add(pad_token, x, pad_len, height, seq_lateral_inds=None):
    pad_shape0 = 1 if pad_token.shape[0] == 1 else x.shape[0]
    C = x.shape[-1]

    if seq_lateral_inds is not None:
        pad_token = rearrange(pad_token, 'l c n -> l n c')
        pad_token = pad_token.reshape(pad_shape0, pad_len, 1, C) # 1, 3, C
        pad_token = pad_token.expand(
            pad_token.shape[0], pad_len, 
            len(seq_lateral_inds)//pad_len, C).reshape(pad_shape0, -1, C)
        x[:, seq_lateral_inds, :] += pad_token
        return x

    x = rearrange(x, 'b (h w) c -> b c h w', h=height)
    pad_token = pad_token.reshape(pad_shape0, C, -1, pad_len) # 1, C, 1, 3
    h_pad_token = pad_token # B, C, 1, 3
    v_pad_token = rearrange(pad_token, 'b c l p -> b c p l') # B, C, 3, 1

    x_new = x.clone()
    x_new[:, :, :, -pad_len:] += h_pad_token
    x_new[:, :, -pad_len:, :-pad_len] += v_pad_token
    x = x_new
    x = rearrange(x, 'b c h w -> b (h w) c')
    return x


def lateral_token_pad(pad_token, x, pad_len, height):
    C = x.shape[-1]
    x = rearrange(x, 'b (h w) c -> b c h w', h=height)
    pad_shape0 = 1 if pad_token.shape[0] == 1 else x.shape[0]
    pad_token = pad_token.reshape(pad_shape0, C, -1, pad_len) # 1, C, 1, 3
    x = torch.cat(
        [
            x,  # B, C, H, W
            pad_token.expand(*x.shape[:3], pad_len),  # B, C, H, 3
        ], -1
    )
    x = torch.cat(
        [
            x,  # B, C, H, W+3
            pad_token.reshape(pad_shape0, C, pad_len, -1).expand(*x.shape[:2], pad_len, x.shape[-1]),  # B, C, 3, W+3
        ], -2
    )  

    width = x.shape[-1]
    x = rearrange(x, 'b c h w -> b (h w) c')
    L = x.shape[-2] #+ self.extra_len
    return x, width, L

def lateral_token_unpad(img_feats, pad_len, height, ):
    img_feats = rearrange(img_feats, 'b (h w) c -> b c h w', h=height + pad_len)
    img_feats = img_feats[..., :-pad_len]
    paded_feats = img_feats[..., -pad_len:] # b c h w1
    width = img_feats.shape[-1]
    img_feats = img_feats[:, :, :height, :]
    
    img_feats = rearrange(img_feats, 'b c h w -> b (h w) c')
    L = img_feats.shape[-2]
    return img_feats, width, paded_feats

def acc_torch_cat(x_list, dim=0):
    shape_len = len(x_list[0].shape)
    dim = shape_len+dim if dim < 0 else dim
    x_shape = list(x_list[0].reshape(*x_list[0].shape[:dim], x_list[0].shape[dim], -1).shape)
    x_shape[dim] = 0
    new_x_list = []
    for i in x_list:
        x_shape[dim] += i.shape[dim]
        new_x_list.append(
            i.reshape(*i.shape[:dim], i.shape[dim], -1)
        )
    
    ptr = 0
    x = x_list[0].new_empty(*x_shape)
    for i in new_x_list:
        x[..., ptr:ptr+i.shape[dim], :].copy_(i)
        ptr += i.shape[dim]
    
    x = x.reshape(*x.shape[:dim], -1, *x_list[0].shape[dim+1:])
    return x


def acc_index_select(x, dim, index):
    return torch.index_select(x, dim=dim, index=index)

def _init_patterns(num_patches, W, pad_token_len, multi_times=1, pattern_type='base'):
    list_temps=[
        dict(
            list1 = range(num_patches // W + pad_token_len), # new after bug
            list2 = range(W + pad_token_len),
            listA = range(0, 1),
            listB = range(0, 1),
        )
    ]
    scan_pattern_ids = [0, ]

    list_temps.append(
        dict(
            list1 = range(num_patches // W - 1 + pad_token_len, -1, -1),
            list2 = range(W - 1 + pad_token_len, -1, -1),
            listA = range(1-1, -1, -1),
            listB = range(1-1, -1, -1),
        )
    )
    scan_pattern_ids.append( -2 ) 
    
    list_temps.append(
        dict(
            list1 = range(W + pad_token_len),
            list2 = range(num_patches // W + pad_token_len),
            listA = range(0, 1),
            listB = range(0, 1),
        )
    )
    scan_pattern_ids.append( 1 )
    
    list_temps.append(
        dict(
            list1 = range(W - 1 + pad_token_len, -1, -1),
            list2 = range(num_patches // W - 1 + pad_token_len, -1, -1),
            listA = range(1-1, -1, -1),
            listB = range(1-1, -1, -1),
        )
    )
    scan_pattern_ids.append( -1 )

    return list_temps, scan_pattern_ids

def token_dropout(x, drop_rate=0.1, interpolation_num=2, ori_x=None, indices=None, is_reverse=False, training=True,):
    '''
        batchsize may affects results
    '''
    if not training:
        return x, None, None
    
    if not is_reverse:
        with torch.no_grad():
            L = x.shape[-2]
            indices = torch.arange(L, device=x.device).reshape(-1, 1).expand(-1, interpolation_num)
            tokens_changed_types = np.random.choice(2, len(indices), p=[1-drop_rate, drop_rate])
            token_keep_ids = (tokens_changed_types == 0)
            token_duplicate_ids = (tokens_changed_types == 1)
            mask = torch.ones_like(indices, dtype=torch.bool)
            mask[ token_keep_ids, 1:] = False
            mask[ token_duplicate_ids, :] = True
            indices = indices[mask].reshape(-1)
        new_x = x[:, indices, :]
        return new_x, indices, x
    else: 
        unique_indices, unique_indices_reverse = get_unique_first_indices(indices, dim=0)
        ori_x = ori_x.clone()
        ori_x[:, unique_indices, :] = x[:, unique_indices_reverse, :].to(ori_x.dtype)
        return ori_x, None, None

def get_unique_first_indices(x, dim):
    unique, idx, counts = torch.unique(x, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], dtype=x.dtype, device=x.device), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    return unique, first_indicies

class SequenceIndexSet(nn.Module):
    def __init__(self, pad_token_len, multi_times, pattern_type, encoder_start_blk_id, scan_pattern_len, num_layers, sub_sequence_schedule): 
        super().__init__()
        self.pad_token_len = pad_token_len
        self.multi_times = multi_times
        self.pattern_type = pattern_type
        self.x_restored_shape = None

        self.encoder_start_blk_id = encoder_start_blk_id
        self.scan_pattern_len = scan_pattern_len
        self.num_layers = num_layers
        self.sub_sequence_schedule = sub_sequence_schedule
    
    def _init_sequence_idx_variables(self, height, width):
        pad_token_len = self.pad_token_len
        multi_times = self.multi_times
        H, W = height, width
        pattern_type = self.pattern_type

        encoder_start_blk_id = self.encoder_start_blk_id
        scan_pattern_len = self.scan_pattern_len
        num_layers = self.num_layers

        num_patches = H * W

        init_pattern_indices_func_type = _init_base_indices

        elem_init_position_in_x_seq, x_seq_position_with_img_arrange, pad_indices = init_pattern_indices_func_type(
            H, W, pad_token_len, dtype=torch.float32, multi_times=multi_times)

        pre_seq_idx = []
        post_seq_idx, flip_patterns, absolute_patterns, scan_pattern_chain = self._init_scan_patterns(
            pad_token_len, W, num_patches,
            encoder_start_blk_id, 
            elem_init_position_in_x_seq,
            scan_pattern_len,
            multi_times, 
            pattern_type,
            num_layers,
            is_random_patterns=False,
        )
        self.x_restored_shape = len(post_seq_idx)

        pre_seq_idx = torch.tensor(pre_seq_idx, dtype=torch.long)
        post_seq_idx = torch.tensor(post_seq_idx, dtype=torch.long)
        if 'reverse_single' in self.sub_sequence_schedule:
            pre_seq_idx = pre_seq_idx.flip(0)
            
        
        self.register_buffer("x_seq_position_with_img_arrange", x_seq_position_with_img_arrange, persistent=False)
        self.register_buffer("pad_indices", pad_indices, persistent=False)
        self.register_buffer("pre_seq_idx", pre_seq_idx, persistent=False)
        self.register_buffer("post_seq_idx", post_seq_idx, persistent=False)
        self.elem_init_position_in_x_seq = elem_init_position_in_x_seq
        self.flip_patterns = flip_patterns
        self.scan_pattern_chain = scan_pattern_chain
        self.absolute_patterns = absolute_patterns
    
    def _init_scan_patterns(
        self, pad_token_len, W, num_patches, 
        encoder_start_blk_id, 
        elem_init_position_in_x_seq,
        scan_pattern_len,
        multi_times, 
        pattern_type,
        num_layers,
        is_random_patterns=False,
    ):

        H = num_patches // W
        absolute_patterns = [] 
        flip_patterns = [] 
        
        list_temps, scan_pattern_ids = _init_patterns(
            num_patches, W, pad_token_len, multi_times=multi_times, pattern_type=pattern_type)
        list_temps = list_temps[ : scan_pattern_len + 1]
        scan_pattern_ids = scan_pattern_ids[ : scan_pattern_len + 1]

        prime_ids = get_multi_times_seq_different_order(
            **list_temps[0],
            elem_init_position_in_x_seq=elem_init_position_in_x_seq, 
            ij_temp=scan_pattern_ids[0],
        )

        if is_random_patterns:
            rand_ids = np.random.choice(
                len(list_temps), 
                num_layers - encoder_start_blk_id - 1,
                replace=True
            )
            rand_ids = [0] * encoder_start_blk_id + rand_ids.tolist() + [0]
            list_temps = [ list_temps[i] for i in rand_ids ] 
            scan_pattern_ids = [ scan_pattern_ids[i] for i in rand_ids ]

        cnt = 0
        assert cnt == 0, 'last pattern must be the forward'
        cnt_list = []
        last_half_cnt_list = []
        mid_layer_idx = encoder_start_blk_id + (num_layers - encoder_start_blk_id)//2

        scan_pattern_chain = [ (0, 0) for _ in range(0, num_layers) ]
        last_scan_pattern = 0
        for layer_idx in range(0, num_layers):
            last_scan_pattern = scan_pattern_ids[cnt]
            if layer_idx < encoder_start_blk_id:
                cnt = 0
                list_temp, ij_temp = list_temps[cnt], scan_pattern_ids[cnt]
                cnt_list.append(cnt)
            elif layer_idx == num_layers - 1:
                cnt = 0
                list_temp, ij_temp = list_temps[cnt], scan_pattern_ids[cnt]
                cnt_list.append(cnt)
            elif layer_idx < mid_layer_idx:
                cnt = (cnt + 1) % len(list_temps)
                list_temp, ij_temp = list_temps[cnt], scan_pattern_ids[cnt]
                cnt_list.append(cnt)
            elif (layer_idx == mid_layer_idx):

                if (num_layers - encoder_start_blk_id) % 2 == 0:
                    cnt = (cnt -1) % len(list_temps)
                else:
                    cnt = (cnt) % len(list_temps)
                
                list_temp, ij_temp = list_temps[cnt], scan_pattern_ids[cnt]
                cnt_list.append(cnt)
            else:
                cnt = (cnt - 1) % len(list_temps)
                list_temp, ij_temp = list_temps[cnt], scan_pattern_ids[cnt]
                cnt_list.append(cnt)
            
            cur_scan_pattern = ij_temp
            
            seq_idx = get_multi_times_seq_different_order(
                **list_temp,
                elem_init_position_in_x_seq=elem_init_position_in_x_seq, 
                ij_temp=ij_temp,
            )
            
            flip_patterns.append(
                get_relative_permutation(
                    absolute_patterns[-1] if len(absolute_patterns) > 0 else prime_ids, 
                    seq_idx, 
                )
            )
            absolute_patterns.append(
                seq_idx 
            )

            scan_pattern_chain[layer_idx] = (last_scan_pattern, cur_scan_pattern)
        
        assert cur_scan_pattern == 0, 'last must be 0 pattern for (1) now content indices '
        return prime_ids, flip_patterns, absolute_patterns, scan_pattern_chain