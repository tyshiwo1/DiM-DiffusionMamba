import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image
from absl import logging
import copy
import datetime
import math

import einops
from einops import rearrange

from mamba_attn_diff.utils.init_weights import init_embedders, initialize_weights, init_adaLN_modulation_layers, _init_weight_norm_fc_conv, pos_embed_inteplot

def get_str_time():
    return str(datetime.datetime.now()).replace(':', '_').replace('.', '_').replace('-', "_").replace(' ', '_')


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})

def load_pretrained_model(model, pretrained_path, **kwargs):
    unmatched_module_names = []
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')

    last_blk_idx = 0
    for k in pretrained_dict.keys():
        if k.split('.')[0] == 'transformer_blocks' and k.split('.')[1].isdigit(): # transformer_blocks.38.attn1.D
            last_blk_idx = max(last_blk_idx, int(k.split('.')[1]))
    
    logging.info(f'last_blk_idx: {last_blk_idx}')

    current_model_dict = model.state_dict()

    new_state_dict = dict()
    for k in current_model_dict.keys():
        mapped_k = k 
        if mapped_k.split('.')[0] == 'transformer_blocks' and mapped_k.split('.')[1].isdigit():
            layer_idx = int(mapped_k.split('.')[1])

        if k not in pretrained_dict.keys():
            logging.info(f'key {k} not in pretrained_dict')
            unmatched_module_names.append(k)
            continue

        elif current_model_dict[k].size() != pretrained_dict[mapped_k].size():

            # if ('pos_embed' in mapped_k) or ('additional_embed' in mapped_k): # pos_embed for reproducing uvit
            if ('additional_embed' in mapped_k):
                
                if hasattr(model, 'extras'):
                    extra_len = model.extras
                elif hasattr(model, 'extra_len'):
                    extra_len = model.extra_len 
                else:
                    extra_len = 0
                
                logging.info(f'interpolating pos_embed for key {mapped_k}, {extra_len}')
                
                pretrained_pos_embed = pos_embed_inteplot(
                    current_model_dict[k], pretrained_dict[mapped_k], extra_len=extra_len)
                new_state_dict.update(
                    {k: pretrained_pos_embed}
                )
            elif 'pos_embed.proj.weight' in mapped_k:
                logging.info(f'conv pos_embed.proj.weight for key {mapped_k}')
                new_conv_kernel = current_model_dict[mapped_k]
                old_conv_kernel = pretrained_dict[mapped_k] # Cin, Cout, Kernel, Kernel
                
                old_conv_kernel = old_conv_kernel[..., 0, 0].unsqueeze(-1).unsqueeze(-1)

                old_conv_kernel = old_conv_kernel.expand_as(new_conv_kernel)
                new_state_dict.update(
                    {k: old_conv_kernel.to(current_model_dict[mapped_k].dtype) }
                )
            elif 'proj_out_2' in mapped_k:

                logging.info(f'proj_out_2 for key {mapped_k}')
                w = pretrained_dict[mapped_k]
                new_patch_size = model.patch_size
                out_channels = model.out_channels
                inner_dim = model.inner_dim
                
                if 'weight' in mapped_k:
                    old_patch_size = int((w.shape[0] // out_channels)**0.5)
                    w = w.reshape(old_patch_size, old_patch_size, out_channels, inner_dim)
                    
                    w = w[0, 0].unsqueeze(0).unsqueeze(0)

                    w = w.expand(new_patch_size, new_patch_size, out_channels, inner_dim)
                    w = w.reshape(new_patch_size**2 * out_channels, inner_dim)
                else:
                    w = w.reshape(old_patch_size, old_patch_size, out_channels)
                    
                    w = w[0, 0].unsqueeze(0).unsqueeze(0)

                    w = w.expand(new_patch_size, new_patch_size, out_channels)
                    w = w.reshape(new_patch_size**2 * out_channels)
                
                new_state_dict.update(
                    {k: w.to(current_model_dict[mapped_k].dtype) }
                )
                
            else:
                logging.info(
                    f'size mismatch for key {k}, current_model_dict[k].size()={current_model_dict[k].size()}, \
                    pretrained_dict[k].size()={pretrained_dict[mapped_k].size()}'
                )
                unmatched_module_names.append(k)
        else:
            new_state_dict.update({k: pretrained_dict[mapped_k]})

    model.load_state_dict(new_state_dict, strict=False)

    logging.info(f'load pretrained model from {pretrained_path}')
    return model, unmatched_module_names

def get_nnet(name, pretrained_path=None, is_init_mamba=False, **kwargs):
    if 'Mamba_DiT' in name:
        from mamba_attn_diff.models.mamba_2d import Mamba2DModel as Mamba_DiT
        model = Mamba_DiT(**kwargs)
        # initialize_weights(model)
        init_embedders(model)
        # # # init_adaLN_modulation_layers(model)
        if is_init_mamba:
            model._init_weights()
        # # model = _init_weight_norm_fc_conv(model)

    elif name == 'uvit':
        from libs.uvit import UViT
        model = UViT(**kwargs)
    elif name == 'uvit_t2i':
        from libs.uvit_t2i import UViT
        model = UViT(**kwargs)
    elif 'DiT' in name:
        from diffusers.models.transformer_2d import Transformer2DModel
        model = Transformer2DModel(**kwargs)
        model.__class__ = make_input_sequence_change(model.__class__)
    else:
        raise NotImplementedError(name)

    unmatched_module_names = None
    if pretrained_path is not None and os.path.exists(pretrained_path):
        model, unmatched_module_names = load_pretrained_model(
            model, pretrained_path, **kwargs)

    return model


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def get_optimizer(params, name, model=None, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        return AdamW(params, **kwargs)
    elif name == 'adamw_stop':
        from torch.optim import AdamW
        lr = kwargs.get('lr', 2e-4)
        lr_specified = kwargs.get('lr_specified', lr)
        if 'lr_specified' in kwargs:
            kwargs.pop('lr_specified')

        target_modules_list = kwargs.get('target_modules_list', ["pos_embed", "proj_out_2"])
        if 'target_modules_list' in kwargs:
            kwargs.pop('target_modules_list')

        specified_params = []
        for module_name, param in model.named_parameters():

            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            
            print('Specified weights: ', module_name, param.shape)
            specified_params.append(param)
        
        id_params = [id(p) for p in specified_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_params]
        for p in model.parameters():
            if id(p) not in id_params:
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)

        param_groups = [ 
            {
                'params': specified_params, 
                'lr': lr_specified,
            }
        ]
        return AdamW(param_groups, **kwargs)

    elif name == 'adamw_specified':
        from torch.optim import AdamW
        lr = kwargs.get('lr', 2e-4)
        lr_specified = kwargs.get('lr_specified', lr)
        if 'lr_specified' in kwargs:
            kwargs.pop('lr_specified')

        target_modules_list = kwargs.get('target_modules_list', ["pos_embed", "proj_out_2"])
        if 'target_modules_list' in kwargs:
            kwargs.pop('target_modules_list')
        
        excluded_modules_list = kwargs.get('excluded_modules_list', [])
        if 'excluded_modules_list' in kwargs:
            kwargs.pop('excluded_modules_list')

        specified_params = []
        for module_name, param in model.named_parameters():

            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            
            if any(target_key in module_name for target_key in excluded_modules_list):
                continue
            
            print('Specified weights: ', module_name, param.shape)
            specified_params.append(param)
        
        id_params = [id(p) for p in specified_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_params]

        param_groups = [
            # { 'params': regular_params, }, 
            {
                'params': specified_params, 
                'lr': lr_specified,
            }
        ]
        return AdamW(param_groups, **kwargs)
    elif name == 'lion':
        try:
            from lion_torch import Lion
        except ImportError:
            raise ImportError('Please install lion_torch, `pip install lion-pytorch`')
        
        print('use Lion Optimizer')
        return Lion(params, **kwargs)

    elif name == 'galore':
        try:
            from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
        except ImportError:
            raise ImportError('Please install galore_torch')
        
        lr = kwargs.get('lr', 1e-5)
        rank = kwargs.get('rank', 256)
        update_proj_gap = kwargs.get('update_proj_gap', 200)
        scale = kwargs.get('scale', 0.25)
        proj_type = kwargs.get('proj_type', 'std')
        is_8bit = kwargs.get('is_8bit', False)
        GaLoreOptim = GaLoreAdamW if not is_8bit else GaLoreAdamW8bit

        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            
            if module.weight.shape[0] < rank or module.weight.shape[1] < rank:
                continue
            
            print('enable GaLore for weights in module: ', module_name, module.weight.shape)
            galore_params.append(module.weight)
        
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
        # then call galore_adamw

        param_groups = [
            { 'params': regular_params, }, 
            {
                'params': galore_params, 
                'rank': rank, 
                'update_proj_gap': update_proj_gap, 
                'scale': scale, 
                'proj_type': proj_type,
            }
        ]
        optimizer = GaLoreOptim(param_groups, lr=lr) 
        return optimizer
    else:
        raise NotImplementedError(name)


from torch.optim.lr_scheduler import CosineAnnealingLR
class WarmupCosineAnnealingLR(CosineAnnealingLR):

    def __init__(self, *args, warmup_steps=0, min_lr_rate=0, **kwargs):
        self.warmup_steps = warmup_steps
        self.min_lr_rate = min_lr_rate
        super().__init__(*args, **kwargs)

    def get_lr(self):
        # print(self._step_count , self.warmup_steps, '2')
        if self._step_count < self.warmup_steps:
            return [base_lr * max(self.min_lr_rate, self._step_count / self.warmup_steps)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        else:
            return super().get_lr()

    def _get_closed_form_lr(self):
        # print(self._step_count , self.warmup_steps)
        if self._step_count < self.warmup_steps:
            return [base_lr * max(self.min_lr_rate, self._step_count / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            return super()._get_closed_form_lr()

def customized_lr_scheduler(
    optimizer, 
    warmup_steps=-1, 
    name='customized', 
    T_max=500000, 
    eta_min=1e-6, 
    min_lr_rate=0., 
    steplr_step_size=40000,
    steplr_gamma=0.1,
    **kwargs,
):
    def fn(step):
        if warmup_steps > 0:
            return max(min_lr_rate, min(step / warmup_steps, 1))
        else:
            return max(min_lr_rate, 1)
    
    if name == 'customized':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
    elif name == 'step_lr':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=steplr_step_size, gamma=steplr_gamma)
    else:
        return WarmupCosineAnnealingLR(
            optimizer, warmup_steps=warmup_steps, T_max=T_max, eta_min=eta_min, min_lr_rate=min_lr_rate, )


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'cosine':
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        return customized_lr_scheduler(optimizer, name=name, **kwargs)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def load(self, path, exclude_accelerate=False):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            
            if exclude_accelerate and key in ['optimizer']: # ['optimizer', 'lr_scheduler', 'nnet']:
                continue
            
            if not os.path.exists(os.path.join(path, f'{key}.pth')):
                continue

            if key != 'step' and val is not None:
                val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))
                logging.info(f'load {key} from {path}')


    def resume(self, ckpt_root, step=None, exclude_accelerate=False):
        if not os.path.exists(ckpt_root):
            return None
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path, exclude_accelerate=exclude_accelerate)
        return ckpt_path

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def initialize_train_state(config, device):
    params = []

    pretrained_path = config.nnet.get('pretrained_path', None)
    old_pretrained_path = None
    if pretrained_path is not None:
        old_pretrained_path = pretrained_path
        is_ema_weight = pretrained_path.split('/')[-1].split('.')[0].endswith('_ema')
        if is_ema_weight:
            pretrained_path = pretrained_path.replace('_ema', '')

    nnet = get_nnet(**config.nnet)
    if (pretrained_path is not None) and pretrained_path != old_pretrained_path and os.path.exists(pretrained_path): 
        nnet, _ = load_pretrained_model(
            nnet, pretrained_path, )
    
    for n, p in nnet.named_parameters():
        p.to(torch.float32).requires_grad_(True)
    
    params = [ p for p in nnet.parameters()]

    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')


    optimizer = get_optimizer(params, model=nnet, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples = unpreprocess_fn(sample_fn(mini_batch_size))
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        if accelerator.is_main_process:
            for sample in samples:
                save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


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

def set_forward(model: torch.nn.Module, make_block_fn, class_name="BasicTransformerBlock"):
    for _, module in model.named_modules():
        if isinstance_str(module, class_name): 
            module.__class__ = make_block_fn(module.__class__)
    
    return model

def make_input_sequence_change(block_class):
    class Transformer2DModelTimeFirst(block_class):
        def forward(
            self, 
            hidden_states, 
            timestep, 
            class_labels = None,
            y = None,
            **kwargs
        ):
            # print(hidden_states, timestep, class_labels, y)
            if class_labels is None:
                class_labels = y if y is not None else \
                    torch.zeros_like(timestep).long()+self.config.num_embeds_ada_norm
            
            return super().forward(
                hidden_states, 
                timestep=timestep, 
                class_labels=class_labels, 
                **kwargs
            )

    return Transformer2DModelTimeFirst