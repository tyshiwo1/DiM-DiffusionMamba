import torch
import torch.nn as nn
from absl import logging
import numpy as np
import math
from tqdm import tqdm
import copy


def get_sde(device='cuda', name='vpsde', **kwargs):
    if name == 'vpsde':
        return VPSDE(**kwargs)
    elif name == 'vpsde_cosine':
        return VPSDECosine(**kwargs)
    elif name in ['ddpm', 'ddim', 'heun', 'euler', 'dpmsolver']:
        return DDPM(device=device, name=name, **kwargs)
    else:
        raise NotImplementedError


def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1, schedule='l2'):  # mean of square
    if schedule == 'l2':
        return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)
        # return (a.nan_to_num(0)).pow(2).flatten(start_dim=start_dim).mean(dim=-1) # mamba is easy to nan
    elif schedule == 'l1':
        return a.abs().flatten(start_dim=start_dim).mean(dim=-1)
    elif schedule == 'pseudo_huber':
        c = 0.03 # cifar 10
        return ((a.pow(2)+(c**2)).pow(0.5) - c).flatten(start_dim=start_dim).mean(dim=-1)


def duplicate(tensor, *size):
    return tensor.unsqueeze(dim=0).expand(*size, *tensor.shape)

def extend_t_dims(t, xt):
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    t = t.to(xt.device)
    if t.dim() == 0:
        t = duplicate(t, xt.size(0))
    
    return t

class SDE(object):
    r"""
        dx = f(x, t)dt + g(t) dw with 0 <= t <= 1
        f(x, t) is the drift
        g(t) is the diffusion
    """
    def drift(self, x, t):
        raise NotImplementedError

    def diffusion(self, t):
        raise NotImplementedError

    def cum_beta(self, t):  # the variance of xt|x0
        raise NotImplementedError

    def cum_alpha(self, t):
        raise NotImplementedError

    def snr(self, t):  # signal noise ratio
        raise NotImplementedError

    def nsr(self, t):  # noise signal ratio
        raise NotImplementedError

    def marginal_prob(self, x0, t):  # the mean and std of q(xt|x0)
        alpha = self.cum_alpha(t)
        beta = self.cum_beta(t)
        mean = stp(alpha ** 0.5, x0)  # E[xt|x0]
        std = beta ** 0.5  # Cov[xt|x0] ** 0.5
        return mean, std

    def sample(self, x0, t_init=0):  # sample from q(xn|x0), where n is uniform
        t = torch.rand(x0.shape[0], device=x0.device) * (1. - t_init) + t_init
        mean, std = self.marginal_prob(x0, t)
        eps = torch.randn_like(x0)
        xt = mean + stp(std, eps)
        return t, eps, xt


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20):
        # 0 <= t <= 1
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def drift(self, x, t):
        return -0.5 * stp(self.squared_diffusion(t), x)

    def diffusion(self, t):
        return self.squared_diffusion(t) ** 0.5

    def squared_diffusion(self, t):  # beta(t)
        return self.beta_0 + t * (self.beta_1 - self.beta_0)

    def squared_diffusion_integral(self, s, t):  # \int_s^t beta(tau) d tau
        return self.beta_0 * (t - s) + (self.beta_1 - self.beta_0) * (t ** 2 - s ** 2) * 0.5

    def skip_beta(self, s, t):  # beta_{t|s}, Cov[xt|xs]=beta_{t|s} I
        return 1. - self.skip_alpha(s, t)

    def skip_alpha(self, s, t):  # alpha_{t|s}, E[xt|xs]=alpha_{t|s}**0.5 xs
        x = -self.squared_diffusion_integral(s, t)
        return x.exp()

    def cum_beta(self, t):
        return self.skip_beta(0, t)

    def cum_alpha(self, t):
        return self.skip_alpha(0, t)

    def nsr(self, t):
        return self.squared_diffusion_integral(0, t).expm1()

    def snr(self, t):
        return 1. / self.nsr(t)

    def __str__(self):
        return f'vpsde beta_0={self.beta_0} beta_1={self.beta_1}'

    def __repr__(self):
        return f'vpsde beta_0={self.beta_0} beta_1={self.beta_1}'

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_heun_discrete import HeunDiscreteScheduler
# from diffusers.schedulers.scheduling_edm_euler import EDMEulerScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

class DDPM(SDE):
    def __init__(
        self, 
        *args, 
        name=None,
        beta_end = 0.02,
        beta_schedule = "linear",
        beta_start = 0.0001,  #clip_sample = False, set_alpha_to_one = True,
        num_train_timesteps = 1000,
        prediction_type = "epsilon",
        steps_offset = 0,
        trained_betas = None,
        device='cuda',
        **kwargs,
    ):
        scheduler_type = DDIMScheduler
        if name == 'ddpm':
            scheduler_type = DDPMScheduler
        elif name == 'heun':
            scheduler_type = HeunDiscreteScheduler
        elif name == 'euler':
            scheduler_type = EulerDiscreteScheduler
        elif name == 'dpmsolver':
            scheduler_type = DPMSolverMultistepScheduler

        noise_scheduler = scheduler_type(
            *args,
            num_train_timesteps=num_train_timesteps,
            prediction_type=prediction_type,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule, #clip_sample=clip_sample, set_alpha_to_one=set_alpha_to_one,
            steps_offset = steps_offset,
            trained_betas=trained_betas,
            **kwargs,
        )
        noise_scheduler.set_timesteps(num_train_timesteps, device=device)
        self.noise_scheduler = noise_scheduler
        self.beta_start = beta_start
        self.beta_end = beta_end

    def sample(self, x0, t_init=0):  # sample from q(xn|x0), where n is uniform
        noise_scheduler = self.noise_scheduler
        bsz = x0.shape[0]
        latents = x0

        # add noise
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps, )

        t = timesteps.to(latents.dtype) / 999
        eps = noise
        xt = noisy_latents
        return t, eps, xt

    def __str__(self):
        return f'DDPM beta_start={self.beta_start} beta_1={self.beta_end}'

    def __repr__(self):
        return f'DDPM beta_start={self.beta_start} beta_1={self.beta_end}'

def get_karras_sigmas_timesteps(timesteps, num_inference_steps, alphas_cumprod, device, interpolation_type='linear'):
    sigmas = np.array(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
    log_sigmas = np.log(sigmas)

    if isinstance(timesteps, torch.Tensor):
        timesteps = timesteps.cpu().numpy()

    if interpolation_type == "linear":
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
    elif interpolation_type == "log_linear":
        sigmas = torch.linspace(np.log(sigmas[-1]), np.log(sigmas[0]), num_inference_steps + 1).exp().numpy()

    sigmas = _convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
    timesteps = np.array([_sigma_to_t(sigma, log_sigmas) for sigma in sigmas])

    sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
    sigmas = torch.from_numpy(sigmas).to(device=device)
    # sigmas = torch.cat([sigmas[:1], sigmas[1:-1].repeat_interleave(2), sigmas[-1:]])

    timesteps = torch.from_numpy(timesteps)
    # timesteps = torch.cat([timesteps[:1], timesteps[1:].repeat_interleave(2)])
    timesteps = timesteps.to(device=device)

    return sigmas, timesteps

def _convert_to_karras(in_sigmas: torch.FloatTensor, num_inference_steps) -> torch.FloatTensor:
    """Constructs the noise schedule of Karras et al. (2022)."""

    # Hack to make sure that other schedulers which copy this function don't break
    # TODO: Add this logic to the other schedulers
    sigma_min = None
    sigma_max = None

    sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
    sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

    rho = 7.0  # 7.0 is the value used in the paper
    ramp = np.linspace(0, 1, num_inference_steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas

def _sigma_to_t(sigma, log_sigmas):
    # get log sigma
    log_sigma = np.log(np.maximum(sigma, 1e-10))

    # get distribution
    dists = log_sigma - log_sigmas[:, np.newaxis]

    # get sigmas range
    low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
    high_idx = low_idx + 1

    low = log_sigmas[low_idx]
    high = log_sigmas[high_idx]

    # interpolate sigmas
    w = (low - log_sigma) / (low - high)
    w = np.clip(w, 0, 1)

    # transform interpolation to time range
    t = (1 - w) * low_idx + w * high_idx
    t = t.reshape(sigma.shape)
    return t


class VPSDECosine(SDE):
    r"""
        dx = f(x, t)dt + g(t) dw with 0 <= t <= 1
        f(x, t) is the drift
        g(t) is the diffusion
    """
    def __init__(self, s=0.008):
        self.s = s
        self.F = lambda t: torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        self.F0 = math.cos(s / (1 + s) * math.pi / 2) ** 2

    def drift(self, x, t):
        ft = - torch.tan((t + self.s) / (1 + self.s) * math.pi / 2) / (1 + self.s) * math.pi / 2
        return stp(ft, x)

    def diffusion(self, t):
        return (torch.tan((t + self.s) / (1 + self.s) * math.pi / 2) / (1 + self.s) * math.pi) ** 0.5

    def cum_beta(self, t):  # the variance of xt|x0
        return 1 - self.cum_alpha(t)

    def cum_alpha(self, t):
        return self.F(t) / self.F0

    def snr(self, t):  # signal noise ratio
        Ft = self.F(t)
        return Ft / (self.F0 - Ft)

    def nsr(self, t):  # noise signal ratio
        Ft = self.F(t)
        return self.F0 / Ft - 1

    def __str__(self):
        return 'vpsde_cosine'

    def __repr__(self):
        return 'vpsde_cosine'


class ScoreModel(object):
    r"""
        The forward process is q(x_[0,T])
    """

    def __init__(self, nnet: nn.Module, pred: str, sde: SDE, T=1):
        assert T == 1
        self.nnet = nnet
        self.pred = pred
        self.sde = sde
        self.T = T
        print(f'ScoreModel with pred={pred}, sde={sde}, T={T}')
        self.train_count = 0

    def predict(self, xt, t, **kwargs):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = t.to(xt.device)
        if t.dim() == 0:
            t = duplicate(t, xt.size(0))
        
        model_output = self.nnet(xt, t * 999, **kwargs)
        model_output = model_output.sample if not isinstance(model_output, torch.Tensor) else model_output
        return model_output  # follow SDE

    def noise_pred(self, xt, t, **kwargs):
        pred = self.predict(xt, t, **kwargs)
        if self.pred == 'noise_pred':
            noise_pred = pred
        elif self.pred == 'x0_pred':
            noise_pred = - stp(self.sde.snr(t).sqrt(), pred) + stp(self.sde.cum_beta(t).rsqrt(), xt)
        else:
            raise NotImplementedError
        return noise_pred

    def x0_pred(self, xt, t, **kwargs):
        pred = self.predict(xt, t, **kwargs)
        if self.pred == 'noise_pred':
            x0_pred = stp(self.sde.cum_alpha(t).rsqrt(), xt) - stp(self.sde.nsr(t).sqrt(), pred)
        elif self.pred == 'x0_pred':
            x0_pred = pred
        else:
            raise NotImplementedError
        return x0_pred

    def score(self, xt, t, **kwargs):
        cum_beta = self.sde.cum_beta(t)
        noise_pred = self.noise_pred(xt, t, **kwargs)
        return stp(-cum_beta.rsqrt(), noise_pred)
    
    def unscale_t_predict(self, xt, t, **kwargs):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = t.to(xt.device)
        if t.dim() == 0:
            t = duplicate(t, xt.size(0))
        
        model_output = self.nnet(xt, t, **kwargs)
        model_output = model_output.sample if not isinstance(model_output, torch.Tensor) else model_output
        return model_output


class ReverseSDE(object):
    r"""
        dx = [f(x, t) - g(t)^2 s(x, t)] dt + g(t) dw
    """
    def __init__(self, score_model):
        self.sde = score_model.sde  # the forward sde
        self.score_model = score_model

    def drift(self, x, t, **kwargs):
        drift = self.sde.drift(x, t)  # f(x, t)
        diffusion = self.sde.diffusion(t)  # g(t)
        score = self.score_model.score(x, t, **kwargs)
        return drift - stp(diffusion ** 2, score)

    def diffusion(self, t):
        return self.sde.diffusion(t)


class ODE(object):
    r"""
        dx = [f(x, t) - g(t)^2 s(x, t)] dt
    """

    def __init__(self, score_model):
        self.sde = score_model.sde  # the forward sde
        self.score_model = score_model

    def drift(self, x, t, **kwargs):
        drift = self.sde.drift(x, t)  # f(x, t)
        diffusion = self.sde.diffusion(t)  # g(t)
        score = self.score_model.score(x, t, **kwargs)
        return drift - 0.5 * stp(diffusion ** 2, score)

    def diffusion(self, t):
        return 0


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


@ torch.no_grad()
def diffusers_denoising(
    score_model, noise_scheduler, x_init, sample_steps, eps=1e-3, 
    do_classifier_free_guidance=False, num_aug_cfg = 2, cfg_weight=1.5, device='cuda', **kwargs,
):
    '''
        TODO: this function has not been tested yet
    '''
    hook_intermediate_out = False
    trained_by_sigma = kwargs.get('trained_by_sigma', False)
    print(f"Diffuers Scheduler with sample_steps={sample_steps}, trained_by_sigma: {trained_by_sigma}")
    nnet = score_model.nnet
    model_input = x_init
    noise_scheduler.set_timesteps(sample_steps, device=device)
    timesteps = noise_scheduler.timesteps
    sigmas = None
    dtype = model_input.dtype 
    if hasattr(noise_scheduler, 'sigmas'):
        sigmas = noise_scheduler.sigmas
    
    for i, timestep in tqdm(enumerate(timesteps)):
        latent_model_input = torch.cat([model_input] * num_aug_cfg) if do_classifier_free_guidance else model_input
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep, )

        if trained_by_sigma:
            t = model_input.new_empty(latent_model_input.shape[0]).fill_( sigmas[i] * 1000 ) 
        else:
            t = model_input.new_empty(latent_model_input.shape[0]).fill_(timestep)
        
        t = t.to(dtype)
        latent_model_input = latent_model_input.to(dtype)
        
        unet_output = nnet(latent_model_input, t, **kwargs)
        model_output = decomp_output(unet_output)

        # perform guidance
        noise_pred_uncond, noise_pred_cond = model_output, model_output
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = model_output.chunk(num_aug_cfg)
            model_output = noise_pred_uncond + cfg_weight * (noise_pred_cond - noise_pred_uncond)
        
        base = model_input
        if hook_intermediate_out:
            next_model_input = model_output
            model_input = next_model_input
        else:
            sch_step_res = noise_scheduler.step(
                model_output, timestep, base, 
            )
            next_model_input = sch_step_res.prev_sample
            model_input = next_model_input
    
    return next_model_input

@ torch.no_grad()
def euler_maruyama(rsde, x_init, sample_steps, eps=1e-3, T=1, trace=None, verbose=False, **kwargs):
    r"""
    The Euler Maruyama sampler for reverse SDE / ODE
    See `Score-Based Generative Modeling through Stochastic Differential Equations`
    """
    assert isinstance(rsde, ReverseSDE) or isinstance(rsde, ODE)
    print(f"euler_maruyama with sample_steps={sample_steps}")
    timesteps = np.append(0., np.linspace(eps, T, sample_steps))
    timesteps = torch.tensor(timesteps).to(x_init)
    x = x_init
    if trace is not None:
        trace.append(x)
    for s, t in tqdm(list(zip(timesteps, timesteps[1:]))[::-1], disable=not verbose, desc='euler_maruyama'):
        drift = rsde.drift(x, t, **kwargs)
        diffusion = rsde.diffusion(t)
        dt = s - t
        mean = x + drift * dt
        sigma = diffusion * (-dt).sqrt()
        x = mean + stp(sigma, torch.randn_like(x)) if s != 0 else mean
        if trace is not None:
            trace.append(x)
        statistics = dict(s=s, t=t, sigma=sigma.item())
        logging.debug(dct2str(statistics))
    
    return x


def LSimple(score_model: ScoreModel, x0, pred='noise_pred', iter_rate=None, **kwargs):
    dtype = x0.dtype
    t, noise, xt = score_model.sde.sample(x0)
    xt = xt.to(dtype)
    noise = noise.to(dtype)
    t = t.to(dtype)
    if pred == 'noise_pred':
        noise_pred = score_model.noise_pred(xt, t, **kwargs)
        return mos(noise - noise_pred)
    elif pred == 'x0_pred':
        x0_pred = score_model.x0_pred(xt, t, **kwargs)
        return mos(x0 - x0_pred)
    else:
        raise NotImplementedError(pred)


def decomp_output(unet_output):
    model_output  = None
    if isinstance(unet_output, torch.Tensor):
        model_output = unet_output
    elif isinstance(unet_output, list):
        if len(unet_output) == 1:
            model_output = unet_output[0]
        else:
            assert NotImplementedError
    else:
        model_output = unet_output.sample
    
    return model_output