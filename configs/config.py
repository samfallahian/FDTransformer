CFG = {
    "data": {
        "env": "tgan",
        "data_path": "dataset/",
        "batch_size": 32
    },
    "model": {
        "generatorUnits": [12, 64, 128, 256, 512, 11],
        "discriminatorUnits": [11, 64, 128, 256, 1],
        "negative_slope": 0.2,
        "dropout": 0.2
    },
    "training": {
        "lr": 0.0005,
        "epoch": 50,
        "optimizer": "Adam",
        "weight_decay": 0.01,
        "has_lr_decay": True,
        "lr_decay_gamma": 0.8,
        "n_classes": 1,
        "n_input": 11,
        "kd_band_width": 5,
    }
}
