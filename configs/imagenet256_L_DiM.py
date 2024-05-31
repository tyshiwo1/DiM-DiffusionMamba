import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.z_shape = (4, 32, 32)

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl_ema.pth'
    )

    # config.gradient_accumulation_steps=2 # 1
    config.max_grad_norm = 1.0

    config.train = d(
        n_steps=300000, # 300000
        batch_size=1024, 
        mode='cond',
        log_interval=10,
        eval_interval=5000,
        save_interval=25000, # 50000
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002, 
        weight_decay=0.03, 
        betas=(0.99, 0.99),
        eps=1e-15,
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000, 
    )

    learned_sigma = False
    latent_size = 32
    in_channels = 4 # 3
    config.nnet = d( 
        name='Mamba_DiT_H_2',
        attention_head_dim=1024//16, num_attention_heads=16, num_layers=49, 
        in_channels=in_channels,
        num_embeds_ada_norm=1000,
        sample_size=latent_size,
        activation_fn="gelu-approximate", #"gelu-approximate",
        attention_bias=True,
        norm_elementwise_affine=False,
        norm_type="ada_norm_single", #"layer_norm",
        out_channels=in_channels*2 if learned_sigma else in_channels,
        patch_size=2, 
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
        pad_token_schedules=['dec_split', 'lateral'],
        is_absorb=False, 
        use_adapter_modules=True,
        sequence_schedule='dilated',
        sub_sequence_schedule=['reverse_single', 'layerwise_cross'],
        pos_encoding_type='learnable', 
        scan_pattern_len=4 -1,
        is_align_exchange_q_kv=False, 
        is_random_patterns=False, 
    ) 
    config.gradient_checkpointing = False

    config.dataset = d(
        name='imagenet',
        path='assets/datasets/ImageNet',
        resolution=256,
        cfg=True,
        p_uncond=0.15, # aligned with u-vit
    )

    config.sample = d(
        sample_steps=50,
        n_samples=50000,
        mini_batch_size=25,  # the decoder is large
        algorithm='dpm_solver',
        cfg=True,
        scale=0.4,
        path=''
    )

    return config
