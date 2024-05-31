import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'

    n_steps = 500000
    config.train = d(
        n_steps=n_steps, 
        batch_size=128,
        mode='uncond',
        log_interval=10,
        eval_interval=5000,
        save_interval=50000,
    )

    lr = 0.0002
    config.optimizer = d(
        name='adamw',
        lr=lr, 
        weight_decay=0.03, 
        betas=(0.99, 0.999),
    )

    config.lr_scheduler = d(
        name='customized', 
        warmup_steps=2500,
    )

    learned_sigma = False
    latent_size = 32
    in_channels = 3
    config.nnet = d(
        name='Mamba_DiT_S_2',
        attention_head_dim=512//1, num_attention_heads=1, num_layers=25, 
        in_channels=in_channels,
        num_embeds_ada_norm=10,
        sample_size=latent_size,
        activation_fn="gelu-approximate",
        attention_bias=True,
        norm_elementwise_affine=False,
        norm_type="ada_norm_single", #"layer_norm"
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
        encoder_start_blk_id=1, #0
        kv_as_one_token_idx=-1,
        num_2d_enc_dec_layers=6,
        pad_token_schedules=['dec_split', 'lateral'],
        is_absorb=False, 
        use_adapter_modules=True,
        sequence_schedule='dilated',
        sub_sequence_schedule=['reverse_single', 'layerwise_cross'],
        pos_encoding_type='learnable',
        scan_pattern_len=3,
        is_align_exchange_q_kv=True,
    )

    config.dataset = d(
        name='cifar10',
        path='assets/datasets/cifar10',
        random_flip=True,
    )

    config.sample = d(
        sample_steps=1000,
        n_samples=50000,
        mini_batch_size=500,
        algorithm='euler_maruyama_sde',
        path=''
    )

    return config
