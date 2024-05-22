import ml_collections
import os

def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    # config.z_shape = (4, 64, 64)

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl_ema.pth'
    )

    config.gradient_accumulation_steps=2 # 1
    config.max_grad_norm = 1.0

    config.train = d(
        n_steps=300000,
        batch_size=128,
        mode='cond',
        log_interval=10,
        eval_interval=1000,
        save_interval=5000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0001,
        weight_decay=0.03, 
        betas=(0.99, 0.99),
        eps=1e-15,
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=1, 
    )

    config.ug_theta = 1  
    config.ug_eta = 0.3 
    config.ug_T = 1

    base_resolution = 512
    patch_size = 2
    multi_times = 3 #2
    resolution = int(base_resolution * multi_times)
    coco_multi_scale = [ resolution, resolution //2 , base_resolution, ]

    learned_sigma = False
    latent_size = resolution // 8
    config.z_shape = (4, latent_size, latent_size)
    in_channels = 4  
    config.nnet = d( 
        name='Mamba_DiT_H_2',
        attention_head_dim=1536//1, num_attention_heads=1, num_layers=49, 
        in_channels=in_channels,
        num_embeds_ada_norm=1000,
        sample_size=latent_size,
        activation_fn="gelu-approximate", 
        attention_bias=True,
        norm_elementwise_affine=False,
        norm_type="ada_norm_single", #"layer_norm",
        out_channels=in_channels*2 if learned_sigma else in_channels,
        patch_size=patch_size, 
        mamba_d_state=16,
        mamba_d_conv=3, 
        mamba_expand=2,
        use_bidirectional_rnn=False,
        mamba_type='enc',
        nested_order=0,
        is_uconnect=True,
        no_ff=True,
        use_conv1d=True,
        is_extra_tokens=True,
        rms=True, 
        use_pad_token=True,
        use_a4m_adapter=True,
        drop_path_rate=0.0, 
        encoder_start_blk_id=1, 
        kv_as_one_token_idx=-1,
        num_2d_enc_dec_layers=6,
        pad_token_schedules=['dec_split', 'rho_pad'], 
        is_absorb=False, 
        use_adapter_modules=True,
        sequence_schedule='dilated',
        sub_sequence_schedule=['reverse_single', 'layerwise_cross'],
        pos_encoding_type='learnable', 
        scan_pattern_len=4 -1,
        is_align_exchange_q_kv=False, 
        is_random_patterns=False,
        pretrained_path = 'workdir/imagenet512_H_DiM_ft/default/ckpts/64000.ckpt/nnet_ema.pth', 
        multi_times=multi_times,
        pattern_type='base', 
        is_freeu=False, freeu_param=(0.15, 0.1, 1.1, 1.2), # (0.3, 0.2, 1.1, 1.2)
        num_patches = [ (i //8 //patch_size)**2 for i in coco_multi_scale],
        is_skip_tune=True, skip_tune_param = (0.82, 1.0), #(0.82, 1.0),
    )
    config.gradient_checkpointing = False

    config.dataset = d(
        name='imagenet512_features',
        path='assets/datasets/imagenet512_features',
        cfg=True,
        p_uncond=0.15
    )
    vis_label = None
    config.sample = d(
        sample_steps=50,
        n_samples=50000,
        mini_batch_size=1,  # the decoder is large
        algorithm='dpm_solver_upsample_g',
        cfg=True,
        scale=3.0, 
        path='workdir/imagenet512_H_DiM_ft/test_tmp/' + (
            str(vis_label) if vis_label is not None else 'all_class'
        ) + '_' + str( resolution ),
        vis_label=vis_label,
    )

    if not os.path.exists(config.sample.path):
        os.makedirs(config.sample.path)

    return config
