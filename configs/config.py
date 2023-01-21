CFG = {
    "data": {
        "data_path": "/mnt/d/sources/data/DL-PTV/Engineered/",
        "batch_size": 16
    },
    "model": {
        "generatorUnits": [10, 128, 256, 7],
        "discriminatorUnits": [10, 256, 128, 1],
        "negative_slope": 0.2,
        "nInput": 8,
        "nUnits": [7, 256, 512, 1024, 10],
        "nOutput": 2
    },
    "training": {
        "lr": 0.001,
        "epoch": 100,
        "optimizer": "RAdam",
        "weight_decay": 0.0001,
        "n_classes" : 3,
        "n_input" : 7
    }
}
