import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

# Upsample Guidance: Scale Up Diffusion Models without Training: https://arxiv.org/abs/2404.01709

def model_wrapper(model, noise_schedule=None, is_cond_classifier=False, classifier_fn=None, classifier_scale=1.,
                  time_input_type='1', total_N=1000, model_kwargs={}):

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        """
        if time_input_type == '0':
            # discrete_type == '0' means that the model is continuous-time model.
            # For continuous-time DPMs, the continuous time equals to the discrete time.
            return t_continuous
        elif time_input_type == '1':
            # Type-1 discrete label, as detailed in the Appendix of DPM-Solver.
            return 1000. * torch.max(t_continuous - 1. / total_N, torch.zeros_like(t_continuous).to(t_continuous))
        elif time_input_type == '2':
            # Type-2 discrete label, as detailed in the Appendix of DPM-Solver.
            max_N = (total_N - 1) / total_N * 1000.
            return max_N * t_continuous
        else:
            raise ValueError("Unsupported time input type {}, must be '0' or '1' or '2'".format(time_input_type))

    def cond_fn(x, t_discrete, y):
        """
        Compute the gradient of the classifier, multiplied with the sclae of the classifier guidance.
        """
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier_fn(x_in, t_discrete)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return classifier_scale * torch.autograd.grad(selected.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if is_cond_classifier:
            y = model_kwargs.get("y", None)
            if y is None:
                raise ValueError("For classifier guidance, the label y has to be in the input.")
            t_discrete = get_model_input_time(t_continuous)
            noise_uncond = model(x, t_discrete, **model_kwargs)
            noise_uncond = noise_uncond.sample if not isinstance(noise_uncond, torch.Tensor) else noise_uncond
            cond_grad = cond_fn(x, t_discrete, y)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            dims = len(cond_grad.shape) - 1
            return noise_uncond - sigma_t[(...,) + (None,) * dims] * cond_grad
        else:
            t_discrete = get_model_input_time(t_continuous)
            model_output = model(x, t_discrete, **model_kwargs)
            model_output = model_output.sample if not isinstance(model_output, torch.Tensor) else model_output
            return model_output

    return model_fn

def _init_taus(snrs, timesteps, m=2):
    scaled_snrs = snrs * m
    approx_snr, taus = torch.min(
        (scaled_snrs.reshape(-1, 1) < snrs.reshape(1, -1)) * snrs.reshape(1, -1), 
        dim=-1,
    )
    taus[approx_snr == 0] = 0
    taus = timesteps[taus]
    return taus

def get_tau(preset_snrs, snr_func, t_continuous, m=2, return_indices=False):
    cur_scaled_snr = snr_func(t_continuous) * (m**2)
    cur_scaled_snr = cur_scaled_snr.reshape(-1, 1)
    preset_snrs = preset_snrs.reshape(1, -1)
    tau = (cur_scaled_snr - preset_snrs).abs().min(dim=-1).indices
    if return_indices:
        return tau
    
    tau = 1 - tau  / (preset_snrs.shape[-1] - 1 )
    tau = tau.clamp(max=t_continuous[0])
    return tau

def _init_timestep_idx_map(timesteps):
    timestep_idx_map = { t.item(): idx for idx, t in enumerate(timesteps) }
    return timestep_idx_map

def make_ufg_nnet(
    cfg_model_func, uncfg_model_func, timesteps, alpha_t_func, beta_t_func, snr_func, 
    N=999, m=2, 
    ug_theta=1, ug_eta = 0.3, ug_T = 1, 
    **kwargs,
):
    # print(timesteps, 'timesteps')
    preset_snrs = snr_func(timesteps)
    
    def ufg_nnet(x, t_continuous):
        
        tau_continuous = get_tau(preset_snrs, snr_func, t_continuous, m=m)
        t = t_continuous * N
        tau = tau_continuous * N
        
        cfg_pred = cfg_model_func(x, timestep=t, use_pos_inteplot=True, **kwargs)
        cfg_pred = cfg_pred.sample if not isinstance(cfg_pred, torch.Tensor) else cfg_pred

        w_t = append_dims(get_wt(t_continuous, theta=ug_theta, eta=ug_eta, T=ug_T), len(x.shape))

        if (w_t > 0).any():
            print([tau[0], t[0], alpha_t_func( t_continuous[0] ), beta_t_func( t_continuous[0] ), t_continuous], ug_theta, ug_eta, ug_T )

            resized_pred_noise = get_resized_input_predicted_noise(
                cfg_model_func, x,
                alpha_t_func( t_continuous ), 
                beta_t_func( t_continuous ), 
                tau, 
                m=m, 
                **kwargs,
            )

            g_pred = cfg_pred + w_t * upsample_guidance( 
                resized_pred_noise,
                cfg_pred, #uncon_pred,
                m=m, 
            )
        else: 
            print(t_continuous )
            g_pred = cfg_pred

        return g_pred
    
    return ufg_nnet

def slide_denoise(model_func, x, t, m=2, y=None, **kwargs):
    x = rearrange(x, 'b c (p h) (q w) -> (b p q) c h w ', p=m, q=m)
    t = t.reshape(-1, 1).expand(-1, m*m).reshape(-1)
    y = y.reshape(-1, 1).expand(-1, m*m).reshape(-1) if y is not None else None
    model_output = model_func(x, timestep=t, y=y, **kwargs)
    model_output = model_output.sample if not isinstance(model_output, torch.Tensor) else model_output
    model_output = rearrange(model_output, '(b p q) c h w -> b c (p h) (q w)', p=m, q=m)
    return model_output

def upsample_guidance(resized_input_predicted_noise, predicted_noise, m=2):
    g_pred = resize_func(
        resized_input_predicted_noise / m - resize_func(
            predicted_noise, scale=1/m,
        ),
        scale = m,
    )
    return g_pred
    
def get_resized_input_predicted_noise(model_func, x, alpha_t, beta_t, tau, m=2, **kwargs):
    p = get_P(alpha_t, beta_t, m=m)
    p = append_dims(p, len(x.shape))
    x = resize_func(x, scale=1/m) / (
        p ** 0.5
    )
    model_output = model_func(x, timestep=tau, use_pos_inteplot=True, **kwargs)
    return model_output.sample if not isinstance(model_output, torch.Tensor) else model_output

def get_wt(t, theta=1, eta=0.6, T=1):
    def h(x):
        return 1*(x >= 0)

    w_t = theta * h(t - (1 - eta)*T)
    return w_t

def get_P(alpha_t, beta_t, m=2, ):
    return alpha_t + beta_t / (m**2)

def resize_func(x, scale):
    if scale > 1:
        return F.interpolate(
            x, 
            scale_factor=scale, 
            mode='nearest',  #align_corners=False,
        )
    elif scale < 1:
        rscale = int(1 / scale)
        x = rearrange(x, 'b c (h p) (w q) -> b c h w (p q)', p=rscale, q=rscale )
        x = x.mean(-1)
        return x
    else:
        return x

def append_dims(x, new_shape_len):
    return x.reshape(*x.shape, *([1]*(new_shape_len - len(x.shape))))