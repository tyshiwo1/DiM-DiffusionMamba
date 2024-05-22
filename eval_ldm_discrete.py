from tools.fid_score import calculate_fid_given_paths
import ml_collections
import numpy as np
import os
import torch
from torch import multiprocessing as mp
import accelerate
import utils
from uvit_datasets import get_dataset
import tempfile
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from absl import logging
import builtins
import libs.autoencoder


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()

def get_skip(alphas, betas):
    N = len(betas) - 1
    skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1:] = alphas[s + 1:].cumprod()
    skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1: t + 1] * skip_alphas[1: t + 1, t]
        skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]
    return skip_alphas, skip_betas


def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


class Schedule(object):  # discrete time
    def __init__(self, _betas, reweight_schedule=None, multi_times=1):
        r""" _betas[0...999] = betas[1...1000]
             for n>=1, betas[n] is the variance of q(xn|xn-1)
             for n=0,  betas[0]=0
        """
        self.reweight_schedule = reweight_schedule
        self.multi_times = multi_times

        self._betas = _betas
        self.betas = np.append(0., _betas)
        
        self.alphas = 1. - self.betas
        self.N = len(_betas)   

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas
        

        if self.reweight_schedule == 'upsampler':
            from mamba_attn_diff.models.upsample_guidance import get_tau
            t_continuous = list(range(1, self.N + 1))
            def snr_func(ts):
                return torch.tensor(self.snr[ts])
            
            adjusted_timesteps_indices = get_tau(
                torch.tensor(self.snr[t_continuous]), snr_func, t_continuous, 
                m = 1./multi_times, 
                return_indices = True,
            )
            adjusted_timesteps = np.array([t_continuous[i] for i in adjusted_timesteps_indices])
            self.adjusted_timesteps = adjusted_timesteps


    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def sample(self, x0):  # sample from q(xn|x0), where n is uniform
        if self.reweight_schedule == 'upsampler':
            select_id = np.random.choice( 2 , (len(x0),))
            ori_n = np.random.choice( len(self.adjusted_timesteps) , (len(x0),))
            n = self.adjusted_timesteps[ ori_n ]
            n = n * select_id + ori_n * (1 - select_id)
            eps = torch.randn_like(x0)
        else:
            n = np.random.choice(list(range(1, self.N + 1)), (len(x0),))
            eps = torch.randn_like(x0)

        xn = stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps)
        return torch.tensor(n, device=x0.device, dtype=x0.dtype), eps, xn.to(x0.dtype)

    def __repr__(self):
        return f'Schedule({self.betas[:10]}..., {self.N})'
    
    def cum_alpha(self, t):
        cum_alphas = self.cum_alphas
        if isinstance(t, torch.Tensor):
            cum_alphas = torch.tensor(cum_alphas, device=t.device, dtype=t.dtype)
        
        t = (t * self.N).long()
        return cum_alphas[t]
    
    def cum_beta(self, t):
        cum_betas = self.cum_betas
        if isinstance(t, torch.Tensor):
            cum_betas = torch.tensor(cum_betas, device=t.device, dtype=t.dtype)
        
        t = (t * self.N).long()
        return cum_betas[t]

    def _snr(self, t):
        snr = self.snr
        if isinstance(t, torch.Tensor):
            snr = torch.tensor(snr, device=t.device, dtype=t.dtype)
        
        t = (t * self.N).long()
        return snr[t]


def evaluate(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=config.output_path)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    dataset = get_dataset(**config.dataset)

    nnet = utils.get_nnet(**config.nnet)
    nnet = accelerator.prepare(nnet)
    logging.info(f'load nnet from {config.nnet_path}')
    if (config.nnet_path is not None) and (config.sample.algorithm != 'dpm_solver_upsample_g'):
        accelerator.unwrap_model(nnet).load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    else:
        accelerator.unwrap_model(nnet)
    
    nnet.eval()

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder.to(device)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def decode_large_batch(_batch):
        decode_mini_batch_size = config.sample.get('mini_batch_size', 50)  # use a small batch size since the decoder is large
        decode_mini_batch_size = min(decode_mini_batch_size, config.sample.get('vae_dec_mini_batch_size', 50))
        decode_mini_batch_size = min(50, decode_mini_batch_size)
        xs = []
        pt = 0
        for _decode_mini_batch_size in utils.amortize(_batch.size(0), decode_mini_batch_size):
            x = decode(_batch[pt: pt + _decode_mini_batch_size])
            pt += _decode_mini_batch_size
            xs.append(x)
        xs = torch.concat(xs, dim=0)
        assert xs.size(0) == _batch.size(0)
        return xs

    if 'cfg' in config.sample and config.sample.cfg and config.sample.scale > 0:  # classifier free guidance
        logging.info(f'Use classifier free guidance with scale={config.sample.scale}')
        def cfg_nnet(x, timestep, y=None, **kwargs):
            _cond = nnet(x, timestep, y=y, **kwargs)
            _uncond = nnet(x, timestep, y=torch.tensor([dataset.K] * x.size(0), device=device), **kwargs)
            _cond = _cond.sample if not isinstance(_cond, torch.Tensor) else _cond
            _uncond = _uncond.sample if not isinstance(_uncond, torch.Tensor) else _uncond
            return _cond + config.sample.scale * (_cond - _uncond)
    else:
        def cfg_nnet(x, timestep, y=None, **kwargs):
            _cond = nnet(x, timestep, y=y, **kwargs)
            _cond = _cond.sample if not isinstance(_cond, torch.Tensor) else _cond
            return _cond

    logging.info(config.sample)
    assert os.path.exists(dataset.fid_stat), f'{dataset.fid_stat} not found'
    logging.info(f'sample: n_samples={config.sample.n_samples}, mode={config.train.mode}, mixed_precision={config.mixed_precision}')

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)

    def sample_z(_n_samples, _sample_steps, **kwargs):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)

        if config.sample.algorithm == 'dpm_solver':
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

            def model_fn(x, t_continuous):
                t = t_continuous * N
                eps_pre = cfg_nnet(x, t, **kwargs)
                return eps_pre

            dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
            _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / N, T=1.)
        
        elif config.sample.algorithm in ['ddpm', 'ddim', 'heun', 'euler', 'dpmsolver', 'edm', 'flow']:

            assert config.get('scheduler', False)
            from sde import get_sde, diffusers_denoising, ScoreModel

            scheduler_config = config.scheduler
            scheduler = get_sde(device=device, **scheduler_config) 
            score_model = ScoreModel(nnet, pred=config.pred, sde=scheduler)

            y = kwargs.get('y', None)
            if y is not None: 
                y = torch.cat([
                    torch.tensor([dataset.K] * y.size(0), device=device),
                    y.to(device),
                ], dim=0)

            kwargs.update(dict(y=y))

            _z = diffusers_denoising(
                score_model=score_model, 
                noise_scheduler=scheduler.noise_scheduler, 
                x_init=_z_init, 
                sample_steps=config.sample.sample_steps, 
                do_classifier_free_guidance=True,
                device=device,
                cfg_weight=1.7, 
                **kwargs,
            )

        elif config.sample.algorithm == 'dpm_solver_upsample_g':
            m = config.nnet.get('multi_times', 1)
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())
            sde_entity = Schedule(
                _betas, 
                reweight_schedule=config.get('reweight_schedule', None), 
                multi_times=m,
            )

            normed_timesteps = torch.arange(1000, dtype=_z_init.dtype, device=device).flip(0) / N
            normed_timesteps[-1] = 1e-5
            
            from mamba_attn_diff.models.upsample_guidance import make_ufg_nnet
            model_fn = make_ufg_nnet(
                cfg_nnet, 
                cfg_nnet, #uncfg_nnet,
                normed_timesteps, 
                sde_entity.cum_alpha, 
                sde_entity.cum_beta, 
                sde_entity._snr, 
                m=m, 
                N=sde_entity.N,
                ug_theta = config.get('ug_theta', 1), 
                ug_eta = config.get('ug_eta', 0.3), 
                ug_T = config.get('ug_T', 1), 
                **kwargs,
            )

            dpm_solver = DPM_Solver(model_fn, noise_schedule)
            _z = dpm_solver.sample(
                _z_init,
                steps=_sample_steps,
                eps=1. / N, T=1.)

        else:
            raise NotImplementedError

        return _z

    def sample_fn(_n_samples): 
        # _z_init = torch.randn(_n_samples, *config.z_shape, device=device)

        if config.train.mode == 'uncond':
            kwargs = dict()
        elif config.train.mode == 'cond':
            kwargs = dict(
                y=dataset.sample_label(
                    _n_samples, 
                    device=device,
                    label=config.sample.get('vis_label', None),
                )
            )
        else:
            raise NotImplementedError
        _z = sample_z(_n_samples, _sample_steps=config.sample.sample_steps, **kwargs)
        return decode_large_batch(_z)

    with tempfile.TemporaryDirectory() as temp_path:
        path = config.sample.path or temp_path
        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
        logging.info(f'Samples are saved in {path}')
        utils.sample2dir(accelerator, path, config.sample.n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess)
        if accelerator.is_main_process:
            fid = calculate_fid_given_paths((dataset.fid_stat, path))
            logging.info(f'nnet_path={config.nnet_path}, fid={fid}')


from absl import flags
from absl import app
from ml_collections import config_flags
import os


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "The nnet to evaluate.")
flags.DEFINE_string("output_path", None, "The path to output log.")


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    evaluate(config)


if __name__ == "__main__":
    app.run(main)
