CFG = {
    "data": {
        "env": "tgan",
        "data_path": "dataset/",
        "batch_size": 128
    },
    "model": {
        # "generatorUnits": [12, 64, 128, 256,  11],
        # "discriminatorUnits": [12, 64, 128, 1],
        "generatorUnits": [7, 64, 128, 256, 5],
        "discriminatorUnits": [7, 64, 128, 256, 1],
        "encoderUnits": [7,128,64,8],
        "decoderUnits": [8,64,128,7],
        "negative_slope": 0.2,
        "dropout": 0.2
    },
    "training": {
        "model_file_name": "cgan",
        "lr": 0.001,
        "epoch": 40,
        "scaled_loss": False,
        "optimizer": "Adam",
        "weight_decay": 0.001,
        "has_lr_decay": True,
        "lr_decay_gamma": 0.8,
        "n_classes": 1,
        "n_input": 5,
        "kd_band_width": 5,
        "optim_beta_min": 0.5,
        "optim_beta_max": 0.999,
        "is_critic": False,
        "contractive_coef": 0.001
    }
}
