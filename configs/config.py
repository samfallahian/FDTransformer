CFG = {
    "data": {
        "data_path": "/mnt/d/sources/data/DL-PTV/Engineered/",
        "batch_size": 16
    },
    "model": {
        "nInput": 8,
        "nUnits": [256, 128, 64],
        "nOutput": 2
    },
    "training": {
        "lr": 0.001,
        "epoch": 100,
        "optimizer": "SGD"
    }
}
