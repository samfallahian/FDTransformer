from executor import executor_tgan, model_handler, executor_cae
from data import data_reader, data_loader
from models import model_cae, model_tgan
from utils import helpers

def main():
    config = helpers.Config()
    cfg = config.from_json("data")
    data_read = data_reader.DataReader()
    X, y, scalar = data_read.load_standardize_data('wind_forecast_2009')
    data_load = data_loader.DataModelLoader(X, y)
    train_loader = data_load.all_data_loader()

    # Creating instance of training class
    trained_cgan_model = executor_tgan.Training(train_loader)
    trained_cae_model = executor_cae.Training(train_loader)

    # Creating instance of model handler for saving and loading results and models
    model_handeling = model_handler.ModelHandler()

    """CAE training process"""
    # cae_model = trained_cae_model.forward()
    # model_handeling.save_model(cae_model, "cae_model_final_wind")

    """Load CAe saved models for transferring weights"""
    model = model_cae.CAE()
    cae_pretrained = model_handeling.load_model(model, "1-cae_model_2_labels_final_wind")
    cae_pretrained.eval()
    pretrained_weight = cae_pretrained.decoding.weight.data

    """CGAN training process"""
    # discriminator_model, generator_model = trained_cgan_model.forward(pretrained_weight)
    # model_handeling.save_model(discriminator_model, "discriminator_model")
    # model_handeling.save_model(generator_model, "generator_model")


    """Generating new data"""
    generator = model_tgan.Generator()
    load_saved_generator = model_handeling.load_model(generator, "2-2labels_generator_model")
    generated_data = model_handeling.generate_data(load_saved_generator, 100, scalar)
    print(generated_data)


    pass


if __name__ == "__main__":
    main()
