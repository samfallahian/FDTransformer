CFG = {
    "data": {
        "env": "tgan",
        "data_path": "dataset/",
        "batch_size": 32
    },
    "model": {
        "generatorUnits": [12, 64, 128, 256, 512, 11],
        "discriminatorUnits": [11, 64, 128, 256, 1],
        "negative_slope": 0.2
    },
    "training": {
        "lr": 0.0001,
        "epoch": 6,
        "optimizer": "Adam",
        "weight_decay": 0.0001,
        "n_classes": 1,
        "n_input": 11
    }
}
