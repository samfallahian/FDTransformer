CFG = {
    "data": {
        "data_path": "dataset/",
        "generation_no": 100,
        "cgan": {
            "batch_size": 2048
        },
        "cae": {
            "batch_size": 512},
    },
    "model": {
        "cgan": {"generatorUnits": [11, 64, 64, 7],
                 "discriminatorUnits": [11, 64, 64, 1],
                 "negative_slope": 0.2,
                 "dropout": 0.3},
        "cae": {"autoencoderUnits": [11, 64, 4, 64, 11],
                "dropout": 0.2}
    },
    "training": {
        "cgan": {
            "model_file_name": "cgan",
            "lr": 0.001,
            "epoch": 40,
            "scaled_loss": False,
            "optimizer": "Adam",
            "weight_decay": 0.001,
            "has_lr_decay": True,
            "lr_decay_gamma": 0.8,
            "kd_band_width": 5,
            "optim_beta_min": 0.5,
            "optim_beta_max": 0.999,
            "is_critic": False,
            "is_transferred": False,
            "gp_lambda": 10
        },
        "cae": {
            "model_file_name": "cae",
            "lr": 0.001,
            "epoch": 100,
            "scaled_loss": False,
            "optimizer": "Adam",
            "weight_decay": 0.001,
            "has_lr_decay": True,
            "lr_decay_gamma": 0.8,
            "optim_beta_min": 0.5,
            "optim_beta_max": 0.999,
            "contractive_coef": 0.0001
        }

    }
}
