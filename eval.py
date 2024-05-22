from tools.fid_score import calculate_fid_given_paths
import ml_collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import multiprocessing as mp
import accelerate
import utils
import sde
from uvit_datasets import get_dataset
import tempfile
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from absl import logging
import builtins

from mamba_attn_diff.models.upsample_guidance import make_ufg_nnet

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
    nnet = accelerator.unwrap_model(nnet)
    nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.eval()
    print(nnet, (config.sample.algorithm if config.get('scheduler', False) else 'dpm_solver'))
    
    def cfg_nnet(x, timestep, y, **kwargs):
        _cond = nnet(x, timestep, y=y, **kwargs)
        _uncond = nnet(x, timestep, y=torch.tensor([dataset.K] * x.size(0), device=device), **kwargs)
        _cond = _cond.sample if not isinstance(_cond, torch.Tensor) else _cond
        _uncond = _uncond.sample if not isinstance(_uncond, torch.Tensor) else _uncond
        return _cond + config.sample.scale * (_cond - _uncond)
    
    def uncfg_nnet(x, timestep, y=None, **kwargs):
        _uncfg = nnet(x, timestep, **kwargs)
        _uncfg = _uncfg.sample if not isinstance(_uncfg, torch.Tensor) else _uncfg
        return _uncfg

    if 'cfg' in config.sample and config.sample.cfg and config.sample.scale > 0:  # classifier free guidance
        logging.info(f'Use classifier free guidance with scale={config.sample.scale}')
        score_model = sde.ScoreModel(cfg_nnet, pred=config.pred, sde=sde.VPSDE())
    else:
        score_model = sde.ScoreModel(uncfg_nnet, pred=config.pred, sde=sde.VPSDE())


    logging.info(config.sample)
    assert os.path.exists(dataset.fid_stat)
    logging.info(f'sample: n_samples={config.sample.n_samples}, mode={config.train.mode}, mixed_precision={config.mixed_precision}')

    def sample_fn(_n_samples):
        if config.sample.algorithm == 'dpm_solver_upsample_g':
            m = 2
            data_shape = tuple([dataset.data_shape[0]] + [ i*m for i in dataset.data_shape[1:] ])
            x_init = torch.randn(_n_samples, *data_shape, device=device)
        else:
            x_init = torch.randn(_n_samples, *dataset.data_shape, device=device)

        if config.train.mode == 'uncond':
            kwargs = dict()
        elif config.train.mode == 'cond':
            kwargs = dict(y=dataset.sample_label(_n_samples, device=device))
        else:
            raise NotImplementedError

        if config.sample.algorithm == 'euler_maruyama_sde':
            rsde = sde.ReverseSDE(score_model)
            return sde.euler_maruyama(rsde, x_init, config.sample.sample_steps, verbose=accelerator.is_main_process, **kwargs)
        elif config.sample.algorithm == 'euler_maruyama_ode':
            rsde = sde.ODE(score_model)
            return sde.euler_maruyama(rsde, x_init, config.sample.sample_steps, verbose=accelerator.is_main_process, **kwargs)
        elif config.sample.algorithm == 'dpm_solver_upsample_g':
            noise_schedule = NoiseScheduleVP(schedule='linear')
            sde_entity = sde.VPSDE()

            normed_timesteps = torch.arange(1000, dtype=x_init.dtype, device=device).flip(0) / 999
            normed_timesteps[-1] = 1e-5
            model_fn = make_ufg_nnet(
                cfg_nnet, 
                uncfg_nnet,
                normed_timesteps, 
                sde_entity.cum_alpha, 
                sde_entity.cum_beta, 
                sde_entity.snr, 
                m=2, 
                **kwargs,
            )

            dpm_solver = DPM_Solver(model_fn, noise_schedule)
            return dpm_solver.sample(
                x_init,
                steps=config.sample.sample_steps,
                eps=1e-4,
                adaptive_step_size=False,
                fast_version=True,
            )
        elif config.sample.algorithm == 'dpm_solver':
            noise_schedule = NoiseScheduleVP(schedule='linear')
            model_fn = model_wrapper(
                score_model.noise_pred,
                noise_schedule,
                time_input_type='0',
                model_kwargs=kwargs
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule)
            return dpm_solver.sample(
                x_init,
                steps=config.sample.sample_steps,
                eps=1e-4,
                adaptive_step_size=False,
                fast_version=True,
            )
        else:
            raise NotImplementedError

    with tempfile.TemporaryDirectory() as temp_path:
        path = config.sample.path or temp_path
        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
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
