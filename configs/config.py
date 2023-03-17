CFG = {
    "data": {
        "data_path": "dataset/",
        "generation_no": 100,
        "cgan": {
            "batch_size": 512
        },
        "cae": {
            "batch_size": 128},
    },
    "model": {
        "cgan": {"generatorUnits": [7, 64, 128, 4],
                 "discriminatorUnits": [7, 64, 64, 1],
                 "negative_slope": 0.2,
                 "dropout": 0.5},
        "cae": {"autoencoderUnits": [7, 64, 4, 64, 7],
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
            # "n_classes": 1,
            "n_input": 4,
            "kd_band_width": 5,
            "optim_beta_min": 0.5,
            "optim_beta_max": 0.999,
            "is_critic": False,
            "is_transferred": False
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
