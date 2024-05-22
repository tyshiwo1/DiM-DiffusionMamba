# DiM: Diffusion Mamba for Efficient High-Resolution Image Synthesis

## Installation

```bash
# create env:
conda env create -f environment.yaml

# if you want to update the env `mamba` with the contents in `~/mamba_attn/environment.yaml`:
conda env update --name mamba --file ~/mamba_attn/environment.yaml --prune

# Compiling Mamba. You need to successfully install casual-conv1d first.
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --user -e .
# If failing to compile, copy the files in ./build/ on another server which has compiled successfully; Maybe --user is necessary.
```

## Preparation Before Training and Evaluation

Please follow [U-ViT](https://github.com/baofff/U-ViT), the same subtitle.

## Checkpoints

|                            Model                             | FID  | training iterations | batch size |
| :----------------------------------------------------------: | :--: | :-----------------: | :--------: |
| [ImageNet 256x256 (Huge/2)](https://drive.google.com/drive/folders/1TTEXKKhnJcEV9jeZbZYlXjiPyV87ZhE0?usp=sharing) | 2.40 |        425K         |    768     |
| [ImageNet 512x512 (fine-tuned Huge/2)](https://drive.google.com/drive/folders/1lupf4_dj4tWCpycnraGrgqh4P-6yK5Xe?usp=sharing) | 3.94 |      Fine-tune      |    240     |

**Note: We use `nnet_ema.pth` for evaluation instead of `nnet.pth`.  `nnet.pth` is the trained model. **

## Evaluation

```sh
# ImageNet 256x256 
accelerate launch --multi_gpu --gpu_ids 0,1,2,3,4,5,6,7 --main_process_port 20039 --num_processes 8 --mixed_precision bf16 ./eval_ldm_discrete.py --config=configs/imagenet256_H_mambaenc_pad_cross_conv_skip1_2scan_vaeema_ada_4scan_test_skiptune.py --nnet_path='workdir/imagenet256_H_mambaenc_pad_cross_conv_skip1_2scan_vaeema_ada_4scan/default/ckpts/425000.ckpt/nnet_ema.pth'

# ImageNet 256x256 (U-ViT-H/2)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm_discrete.py --config=configs/imagenet256_uvit_huge.py

# ImageNet 512x512 (U-ViT-L/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm.py --config=configs/imagenet512_uvit_large.py

# ImageNet 512x512 (U-ViT-H/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm_discrete.py --config=configs/imagenet512_uvit_huge.py

# MS-COCO (U-ViT-S/2)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_t2i_discrete.py --config=configs/mscoco_uvit_small.py

# MS-COCO (U-ViT-S/2, Deep)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_t2i_discrete.py --config=configs/mscoco_uvit_small.py --config.nnet.depth=16
```