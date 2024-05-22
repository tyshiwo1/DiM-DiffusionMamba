import torch
from torch.fft import fftn, fftshift, ifftn, ifftshift


def fourier_filter(x_in: "torch.Tensor", threshold: int, scale: int):
    """Fourier filter as introduced in FreeU (https://arxiv.org/abs/2309.11497).

    This version of the method comes from here:
    https://github.com/huggingface/diffusers/pull/5164#issuecomment-1732638706
    """
    x = x_in
    B, L, C = x.shape

    # Non-power of 2 images must be float32
    if (L & (L - 1)) != 0 :
        x = x.to(dtype=torch.float32)

    # FFT
    x_freq = fftn(x, dim=1)
    x_freq = fftshift(x_freq, dim=1)

    B, L, C = x_freq.shape
    mask = torch.ones((B, L, C), device=x.device)

    crow = L // 2
    mask[..., crow - threshold : crow + threshold, :] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = ifftshift(x_freq, dim=1)
    x_filtered = ifftn(x_freq, dim=1).real

    return x_filtered.to(dtype=x_in.dtype)


def apply_freeu(
    resolution_idx, 
    hidden_states, res_hidden_states,
    s1=0.6, s2=0.4, b1=1.1, b2=1.2,
    encoder_start_blk_id=1,
    num_layers=49,
    extra_len=2,
):
    """Applies the FreeU mechanism as introduced in https:
    //arxiv.org/abs/2309.11497. Adapted from the official code repository: https://github.com/ChenyangSi/FreeU.

    Args:
        resolution_idx (`int`): Integer denoting the UNet block where FreeU is being applied.
        hidden_states (`torch.Tensor`): Inputs to the underlying block.
        res_hidden_states (`torch.Tensor`): Features from the skip block corresponding to the underlying block.
        s1 (`float`): Scaling factor for stage 1 to attenuate the contributions of the skip features.
        s2 (`float`): Scaling factor for stage 2 to attenuate the contributions of the skip features.
        b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
        b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.

        pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)

        for i, up_block_type in enumerate(up_block_types):
            resolution_idx=i,
    """
    # if resolution_idx == encoder_start_blk_id + (num_layers - encoder_start_blk_id)//2 + 0:
    #     # print(resolution_idx)
    #     num_half_channels = hidden_states.shape[-1] // 2
    #     hidden_states[..., :num_half_channels] = hidden_states[..., :num_half_channels] * b1
    #     res_hidden_states = fourier_filter(res_hidden_states, threshold=1, scale=s1, )
    # elif resolution_idx == encoder_start_blk_id + (num_layers - encoder_start_blk_id)//2 + 1:
    #     # print(resolution_idx)
    #     num_half_channels = hidden_states.shape[-1] // 2
    #     hidden_states[..., :num_half_channels] = hidden_states[..., :num_half_channels] * b2
    #     res_hidden_states = fourier_filter(res_hidden_states, threshold=1, scale=s2, )

    if resolution_idx == encoder_start_blk_id + (num_layers - encoder_start_blk_id)//2 + 0:
        s = s1
        b = b1
    elif resolution_idx <= encoder_start_blk_id + (num_layers - encoder_start_blk_id)//2 + 1:
        s = s2
        b = b2
    
    if resolution_idx >= encoder_start_blk_id + (num_layers - encoder_start_blk_id)//2 + 0 and \
       resolution_idx <= encoder_start_blk_id + (num_layers - encoder_start_blk_id)//2 + 1 :

        hidden_mean = hidden_states[:, extra_len:, :].mean(-1).unsqueeze(-1)
        B = hidden_mean.shape[0]
        hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1) 
        hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1)
        hidden_max = hidden_max.unsqueeze(-1).unsqueeze(-1)
        hidden_min = hidden_min.unsqueeze(-1).unsqueeze(-1)
        hidden_mean = (hidden_mean - hidden_min) / (hidden_max - hidden_min)


        hidden_states[:, extra_len:, :] = hidden_states[:, extra_len:, :] * ((b - 1 ) * hidden_mean + 1)
        res_hidden_states[:, extra_len:, :] = fourier_filter(res_hidden_states[:, extra_len:, :], threshold=1, scale=s, ) 



    return hidden_states, res_hidden_states