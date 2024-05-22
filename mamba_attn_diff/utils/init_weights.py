import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

from einops import rearrange

def pos_embed_into_2d(pos_embed, extra_len):
    extra_tokens = pos_embed[:, :extra_len]
    pos_embed = pos_embed[:, extra_len:]
    num_patches = pos_embed.shape[-2]
    H = int(num_patches ** 0.5)
    W = num_patches // H
    assert H * W == num_patches
    pos_embed = pos_embed.reshape(
        pos_embed.shape[0], H, W, -1)
    return pos_embed, extra_tokens

def pos_embed_inteplot(cur_pos_embed, pretrained_pos_embed, extra_len, cur_size=None):
    if cur_pos_embed is not None:
        cur_pos_embed, _ = pos_embed_into_2d(cur_pos_embed, extra_len)
        cur_size = cur_pos_embed.shape[-3:-1]
    
    ori_pretrained_pos_embed = pretrained_pos_embed
    pretrained_pos_embed, extra_tokens = pos_embed_into_2d(pretrained_pos_embed, extra_len)
    if pretrained_pos_embed.shape[-3:-1] == cur_size:
        return ori_pretrained_pos_embed
    
    pretrained_pos_embed = F.interpolate(
        rearrange(pretrained_pos_embed, 'b h w d -> b d h w'),
        size= cur_size,
        mode='bilinear', align_corners=False,
    )
    pretrained_pos_embed = rearrange(pretrained_pos_embed, 'b d h w -> b (h w) d')
    pretrained_pos_embed = torch.cat([extra_tokens, pretrained_pos_embed], dim=-2)
    return pretrained_pos_embed

def _init_weight_norm_fc_conv(model):
    for name, module in model.named_modules():
        if isinstance_str(module, "Conv2d") and ("adapter" in name):
            module.__class__ = make_weight_norm_conv2d_nobias(module.__class__)
            module._init_scale()
        
        if isinstance_str(module, "Linear") and ("adapter" in name): 
            module.__class__ = make_weight_norm_fc_nobias(module.__class__)
            module._init_scale()
    
    return model

def make_weight_norm_conv2d_nobias(block_class):
    class WNConv2d(nn.Conv2d):
        def _init_scale(self):
            self.scale = nn.Parameter(self.weight.new_ones(*self.weight.shape[1:]))
        
        def forward(self, x):

            fan_in = self.weight[0].numel()
            weight = weight_normalize(self.weight) / np.sqrt(fan_in) * self.scale

            return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    return WNConv2d

def make_weight_norm_fc_nobias(block_class):
    class WNLinear(nn.Linear):
        def _init_scale(self):
            self.scale = nn.Parameter(self.weight.new_ones(*self.weight.shape[1:]))
        
        def forward(self, x):

            fan_in = self.weight[0].numel()
            weight = weight_normalize(self.weight) / np.sqrt(fan_in) * self.scale

            return F.linear(x, weight, self.bias)
    
    return WNLinear

def weight_normalize(x, eps=1e-4):
    dim = list(range(1, x.ndim))
    n = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    alpha = np.sqrt(n.numel() / x.numel())
    return x / torch.add(eps, n, alpha=alpha)

def _init_weights_mamba(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def init_embedders(model):

    for name, module in model.named_modules():
        if 'class_embedder' in name.lower():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                # print('class_embedder', module.weight.shape)
        elif 'timestep_embedder' in name.lower():
            if isinstance(module, nn.Linear) :
                nn.init.normal_(module.weight, std=0.02)

def init_adaLN_modulation_layers(model):

    for name, module in model.named_modules():

        if 'blocks' in name.lower() and ('norm' in name.lower() or "adaln_single" in name.lower()):
            if isinstance(module, nn.Linear) :
                nn.init.constant_(module.weight, 0)
                nn.init.constant_(module.bias, 0)

def initialize_weights(model):
    # Initialize transformer layers:
    for name, module in model.named_modules():
        if 'mamba' in name.lower():
            continue
        
        if 'embed' in name.lower():
            continue
        
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    for name, module in model.named_modules():
        if 'pos_embed.proj' in name.lower() or "proj_in" in name.lower():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                w = module.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                nn.init.constant_(module.bias, 0)

    init_embedders(model)
    init_adaLN_modulation_layers(model)

    for name, module in model.named_modules():
        if 'out' in name.lower():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                nn.init.constant_(module.weight, 0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False