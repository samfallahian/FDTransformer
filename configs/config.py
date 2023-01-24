CFG = {
    "data": {
        "env": "cgan",
        "data_path": "dataset/",
        "batch_size": 16
    },
    "model": {
        "generatorUnits": [8, 128, 256, 7],
        "discriminatorUnits": [8, 256, 128, 1],
        # "generatorUnits": [3, 64, 128, 7],
        # "discriminatorUnits": [7, 64, 128, 1],
        "negative_slope": 0.2,
        "nInput": 8,
        "nUnits": [7, 256, 512, 1024, 10],
        "nOutput": 2
    },
    "training": {
        "lr": 0.001,
        "epoch": 100,
        "optimizer": "Adam",
        "weight_decay": 0.0001,
        "n_classes" : 3,
        "n_input" : 7
    }
}
