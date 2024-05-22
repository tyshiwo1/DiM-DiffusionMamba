import ml_collections
import torch
from torch import multiprocessing as mp
from uvit_datasets import get_dataset, multiscale_collate_fn, MultiscaleDistSampler, MultiscaleBatchSampler
from torchvision.utils import make_grid, save_image
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
import tempfile
from tools.fid_score import calculate_fid_given_paths
from absl import logging
import builtins
import os
import wandb
import libs.autoencoder
import numpy as np
from functools import partial 

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


def mos(a, start_dim=1, w=None):  # mean of square
    ans = a.pow(2).flatten(start_dim=start_dim)
    if w is not None:
        w = w.reshape(-1, 1)
        ans = ans * w

    return ans.mean(dim=-1)


class Schedule(object):  # discrete time
    def __init__(self, _betas, reweight_schedule=None, multi_times=1, device='cuda'):
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

        # skip_alphas[s, t] = alphas[s + 1: t + 1].prod()
        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

        logging.info(f'snr {self.snr}')
        

        if self.reweight_schedule in ['upsampler', ]:
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

            print('adjusted_timesteps ', adjusted_timesteps )

        if self.reweight_schedule == 'upsampler':
            logging.info('upsampler')
        
        self.snr = torch.tensor(self.snr).to(device)


    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def sample(self, x0):  # sample from q(xn|x0), where n is uniform
        if self.reweight_schedule == 'upsampler':
            select_id = np.random.choice( 2 , (len(x0),))
            ori_n = np.random.choice( len(self.adjusted_timesteps) , (len(x0),))
            n = self.adjusted_timesteps[ ori_n ]
            n = n * select_id + ori_n * (1 - select_id)
            eps = torch.randn_like(x0)
            xn = stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps)
        elif self.reweight_schedule == 'continuous':
            n = np.random.rand(len(x0)) * self.N 
            eps = torch.randn_like(x0)

            n_floor = np.floor(n).astype(np.int64)
            n_ceil = np.ceil(n).astype(np.int64)
            n_rate = (n - n_floor) / (n_ceil - n_floor + 1e-6)
            inter_cum_alphas = self.cum_alphas[n_floor] + n_rate * (self.cum_alphas[n_ceil] - self.cum_alphas[n_floor])
            inter_cum_betas = self.cum_betas[n_floor] + n_rate * (self.cum_betas[n_ceil] - self.cum_betas[n_floor])

            xn = stp(inter_cum_alphas ** 0.5, x0) + stp(inter_cum_betas ** 0.5, eps)
        else:
            n = np.random.choice(list(range(1, self.N + 1)), (len(x0),))
            eps = torch.randn_like(x0)
            xn = stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps)
        
        return torch.tensor(n, device=x0.device, dtype=x0.dtype), eps, xn.to(x0.dtype)

    def __repr__(self):
        return f'Schedule({self.betas[:10]}..., {self.N})'


def LSimple(x0, nnet, schedule, is_snr=False, **kwargs):
    n, eps, xn = schedule.sample(x0)  # n in {1, ..., 1000}
    eps_pred = nnet(xn, n, **kwargs)
    eps_pred = eps_pred.sample if not isinstance(eps_pred, torch.Tensor) else eps_pred
    w = None
    if is_snr:
        n = n.to(torch.long).clamp(min=0, max=len(schedule.snr))
        snr = schedule.snr[n]
        if isinstance(snr, np.ndarray):
            snr = torch.from_numpy(snr).type_as(x0).to(x0.device)
        
        w = snr.clamp(max=5)

    return mos(eps - eps_pred, w=w)


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    if config.nnet.get('latent_size', False):
        assert config.z_shape[-1] == config.nnet.latent_size

    mp.set_start_method('spawn')
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    try:
        deepspeed_plugin = accelerate.DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=gradient_accumulation_steps)
        accelerator = accelerate.Accelerator(
            deepspeed_plugin=deepspeed_plugin
        )
        logging.info('Using deepspeed accelerator')
    except:
        logging.info('Using default accelerator')
        accelerator = accelerate.Accelerator(
            gradient_accumulation_steps = gradient_accumulation_steps,
        )
        
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.init(dir=os.path.abspath(config.workdir), project=f'uvit_{config.dataset.name}', config=config.to_dict(),
                   name=config.hparams, job_type='train', mode='offline')
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    dataset = get_dataset(**config.dataset)
    assert os.path.exists(dataset.fid_stat), "Please download the fid stat file first {}".format(dataset.fid_stat)
    train_dataset = dataset.get_split(split='train', labeled=config.train.mode == 'cond')
    
    if isinstance(train_dataset.resolution, list) or isinstance(train_dataset.resolution, tuple):
        data_sampler = MultiscaleDistSampler(
            resolution=train_dataset.resolution, 
            mini_batch_size=mini_batch_size,
            dataset=train_dataset, 
            shuffle=True, 
            num_replicas=accelerator.num_processes, 
            rank=accelerator.process_index,
        )
        batch_data_sampler = MultiscaleBatchSampler(
            sampler=data_sampler, 
            resolution=train_dataset.resolution,
            batch_size=mini_batch_size,
            drop_last=True,
        )
        train_dataset_loader = DataLoader(
            train_dataset, batch_size=mini_batch_size, drop_last=True,
            sampler=data_sampler, #batch_sampler=batch_data_sampler,
            num_workers=config.train.get('num_workers', 8), #8, 
            pin_memory=True, 
            persistent_workers=(True if config.train.get('num_workers', 8) > 0 else False),
        )
    else:
        train_dataset_loader = DataLoader(
            train_dataset, batch_size=mini_batch_size, drop_last=True, shuffle=True, 
            num_workers=config.train.get('num_workers', 8), #8, 
            pin_memory=True, 
            persistent_workers=(True if config.train.get('num_workers', 8) > 0 else False),
        )

    train_state = utils.initialize_train_state(config, device)
    ckpt_path = train_state.resume(config.ckpt_root, exclude_accelerate=True)
    
    if hasattr(train_state.nnet, 'enable_gradient_checkpointing') and config.get('gradient_checkpointing', False):
        train_state.nnet.enable_gradient_checkpointing()
    
    # nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
    #     train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader)
    nnet, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.optimizer, train_dataset_loader)
    nnet_ema = train_state.nnet_ema.to(device).to(torch.float32)
    dtype = torch.bfloat16

    lr_scheduler = train_state.lr_scheduler
    if ckpt_path is not None:
        # train_state.resume(config.ckpt_root)
        accelerator.load_state(ckpt_path)

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder.to(device)
    autoencoder.requires_grad_(False) 

    @ torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @ torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()

    _betas = stable_diffusion_beta_schedule()
    _schedule = Schedule(
        _betas, 
        reweight_schedule=config.get('reweight_schedule', None), 
        multi_times=config.nnet.get('multi_times', 1),
        device=device,
    )
    logging.info(f'use {_schedule}')

    max_grad_norm = config.get('max_grad_norm', False)
    logging.info(f'Using max_grad_norm={max_grad_norm}')

    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()

        y = _batch[1] if config.train.mode == 'cond' else None
        _z = _batch[0] if config.train.mode == 'cond' else _batch
        if 'nested_feature' in config.dataset.name:
            _z = autoencoder.scale_factor * _z
        elif 'feature' in config.dataset.name:
            _z = autoencoder.sample(_z)
        else:
            _z = encode(_z)
        
        _z = _z.to(dtype)

        loss = LSimple(_z, nnet, _schedule, is_snr=config.train.get('is_snr', False), y=y)

        _metrics['loss'] = accelerator.gather(loss.detach()).mean()
        accelerator.backward(loss.mean())
        if max_grad_norm:
            params_to_clip = filter(lambda p: p.requires_grad, nnet.parameters())
            accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    def dpm_solver_sample(_n_samples, _sample_steps, **kwargs):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        def model_fn(x, t_continuous):
            t = t_continuous * _schedule.N
            eps_pre = nnet_ema(x, t, **kwargs)
            eps_pre = eps_pre.sample if not isinstance(eps_pre, torch.Tensor) else eps_pre
            return eps_pre

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1.)
        return decode(_z)

    def eval_step(n_samples, sample_steps):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}'
                     f'mini_batch_size={config.sample.mini_batch_size}')

        def sample_fn(_n_samples):
            if config.train.mode == 'uncond':
                kwargs = dict()
            elif config.train.mode == 'cond':
                kwargs = dict(y=dataset.sample_label(_n_samples, device=device))
            else:
                raise NotImplementedError
            return dpm_solver_sample(_n_samples, sample_steps, **kwargs)


        with tempfile.TemporaryDirectory() as temp_path:
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
            utils.sample2dir(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess)

            _fid = 0
            if accelerator.is_main_process:
                _fid = calculate_fid_given_paths((dataset.fid_stat, path))
                logging.info(f'step={train_state.step} fid{n_samples}={_fid}')
                with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                    print(f'step={train_state.step} fid{n_samples}={_fid}', file=f)
                wandb.log({f'fid{n_samples}': _fid}, step=train_state.step)
            _fid = torch.tensor(_fid, device=device)
            _fid = accelerator.reduce(_fid, reduction='sum')

        return _fid.item()

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        metrics = train_step(batch)

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        if train_state.step % config.train.eval_interval == 0:
            torch.cuda.empty_cache()
            if accelerator.is_main_process: 
                logging.info('Save a grid of images...')
            
            with torch.no_grad():
                if config.train.mode == 'uncond':
                    _n_samples = 5 if config.get('z_shape', False) and (config.z_shape[-1] >= 128) else 5*10
                    samples_list = []
                    for i in range(0, _n_samples, config.sample.mini_batch_size):
                        samples = dpm_solver_sample(_n_samples=config.sample.mini_batch_size, _sample_steps=50)
                        samples_list.append(samples)
                    
                    samples_list = torch.cat(samples_list, dim=0)
                    samples = samples_list[:_n_samples]
                    
                elif config.train.mode == 'cond':
                    _n_samples = 5*10
                    y = einops.repeat(torch.arange(5, device=device) % dataset.K, 'nrow -> (nrow ncol)', ncol=10)
                    samples = dpm_solver_sample(_n_samples=_n_samples, _sample_steps=50, y=y)
                else:
                    raise NotImplementedError
                
                if accelerator.is_main_process:
                    samples = make_grid(dataset.unpreprocess(samples), min(10, _n_samples) )
                    save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
                    wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
            
            torch.cuda.empty_cache()
        
        accelerator.wait_for_everyone()

        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Save and eval checkpoint {train_state.step}...')

            save_path = os.path.join(config.ckpt_root, f'{train_state.step}.ckpt')
            if accelerator.local_process_index == 0:
                # train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
                os.makedirs(save_path, exist_ok=True)
                torch.save(train_state.step, os.path.join(save_path, 'step.pth'))
                torch.save(nnet_ema.state_dict(), os.path.join(save_path, 'nnet_ema.pth'))
            
            accelerator.save_state(output_dir=save_path)

            accelerator.wait_for_everyone()
            fid = eval_step(n_samples=min(10000, config.sample.get('intermidiate_n_samples', 10000) ), sample_steps=50)  # calculate fid of the saved checkpoint
            step_fid.append((train_state.step, fid))
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_fid: {step_fid}')
    step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
    logging.info(f'step_best: {step_best}')
    train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt'))
    del metrics
    accelerator.wait_for_everyone()
    eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps)



from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'default'
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    config.workdir = FLAGS.workdir or os.path.join('workdir', config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)
