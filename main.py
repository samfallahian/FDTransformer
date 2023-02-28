from executor import executor_tgan, model_loader, executor_cae
from data import data_reader, data_loader
from models import model_tgan
from utils import helpers

def main():
    config = helpers.Config()
    cfg = config.from_json("data")
    data_read = data_reader.DataReader()
    X, y = data_read.load_standardize_data('wind_forecast_2009')
    data_load = data_loader.DataModelLoader(X, y)
    train_loader = data_load.all_data_loader()

    model_training_cgan = executor_tgan.Training(train_loader)
    # model_training_cae = executor_cae.Training(train_loader)
    model_handeling = model_loader.ModelLoader()
    """CGAN"""
    discriminator_model, generator_model = model_training_cgan.forward()
    model_handeling.save_model(discriminator_model, generator_model)
    """CAE"""
    
    # cae_model = model_training_cae.forward()
    """Generating new data"""
    # generator = model_tgan.Generator()
    # load_saved_generator = model_handeling.load_model(generator, "2023-02-05 15:05:24.747473-cgan-generator_model")
    # generated_data = model_handeling.generate_data(load_saved_generator)
    # print(generated_data)


    pass


if __name__ == "__main__":
    main()
