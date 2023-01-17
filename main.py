from executor import executor_ann
from data import data_reader, data_loader

data_read = data_reader.DataReader()
X, y = data_read.load_standardize_data('test')
data_load = data_loader.DataModelLoader(X, y)
train_loader, test_loader = data_load.train_test_data_loader()

model_training = executor_ann.Training(train_loader, test_loader)

model_training.exec()
