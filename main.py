from executor import executor_cgan, executor_tgan
from data import data_reader, data_loader
from utils import helpers

def main():
    config = helpers.Config()
    cfg = config.from_json("data")
    data_read = data_reader.DataReader()
    X, y = data_read.load_standardize_data('test_500')
    data_load = data_loader.DataModelLoader(X, y)
    train_loader = data_load.all_data_loader()
    if cfg.env == "cgan":
        model_training = executor_cgan.Training(train_loader)
    else:
        model_training = executor_tgan.Training(train_loader)

    model_training.forward()






    # train_loader, test_loader = data_load.train_test_data_loader() # all_data_loader
    #
    # model_training = executor_ann.Training(train_loader, test_loader)
    #
    # model_training.exec()
    pass


if __name__ == "__main__":
    main()
