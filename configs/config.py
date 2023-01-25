CFG = {
    "data": {
        "env": "tgan",
        "data_path": "dataset/",
        "batch_size": 32
    },
    "model": {
        "generatorUnits": [12, 64, 128, 256, 512, 11],
        "discriminatorUnits": [11, 64, 32, 16, 8, 1],
        # "generatorUnits": [3, 64, 128, 7],
        # "discriminatorUnits": [7, 64, 128, 1],
        "negative_slope": 0.2,
        "nInput": 8,
        "nUnits": [7, 256, 512, 1024, 10],
        "nOutput": 2
    },
    "training": {
        "lr": 0.0001,
        "epoch": 100,
        "optimizer": "Adam",
        "weight_decay": 0.0001,
        "n_classes": 3,
        "n_input": 8,
        "n_generator_input": 7
    }
}
