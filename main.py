from executor import executor_tgan, model_handler, executor_cae
from data import data_reader, data_loader
from models import model_cae, model_tgan
from utils import helpers
import pandas as pd


def main():
    config = helpers.Config()
    cfg = config.from_json("data")
    data_read = data_reader.DataReader()
    # X, y, scalar = data_read.load_standardize_data('3p6-selected')
    X, y, scalar = data_read.load_standardize_data_test('3p6-selected-test')
    X_df = pd.DataFrame(X, columns=['vx', 'vy', 'vz', 'px', 'py', 'pz', 'distance'])
    y_df = pd.DataFrame(y, columns=['x','y','z','time'])
    df = pd.concat([X_df, y_df], axis=1)
    df.to_csv("logs/normalized_input.csv", index=False)
    # data_load = data_loader.DataModelLoader(X, y, "cae")
    # data_load = data_loader.DataModelLoader(X, y, "cgan")
    # train_loader = data_load.all_data_loader()

    # Creating instance of training class
    # trained_cgan_model = executor_tgan.Training(train_loader)
    # trained_cae_model = executor_cae.Training(train_loader)

    # Creating instance of model handler for saving and loading results and models
    model_handeling = model_handler.ModelHandler()

    """CAE training process"""
    # cae_model = trained_cae_model.forward()
    # model_handeling.save_model(cae_model, "cae_model_ptv")

    """Load CAe saved models for transferring weights"""
    # model = model_cae.CAE()
    # cae_pretrained = model_handeling.load_model(model, "5_cae_model_ptv")
    # cae_pretrained.eval()
    # pretrained_weight = cae_pretrained.decoding.weight.data

    """CGAN training process"""
    # discriminator_model, generator_model = trained_cgan_model.forward(pretrained_weight)
    # # discriminator_model, generator_model = trained_cgan_model.forward()
    # model_handeling.save_model(discriminator_model, "discriminator_model")
    # model_handeling.save_model(generator_model, "generator_model")

    """Generating new data"""
    generator = model_tgan.Generator()
    load_saved_generator = model_handeling.load_model(generator, "8_gen_ptv")
    labels = pd.read_csv("dataset/3p6-selected-labels-test.csv", index_col=False)
    # labels = pd.read_pickle("dataset/3p6-relative_labels_test.pkl", compression="zip")
    generated_data = model_handeling.generate_data(load_saved_generator, labels, scalar)
    ### df_result = pd.DataFrame(generated_data, columns=["time_frame", "hrs", "farm", "u", "v", "ws", "wd"])
    df_result = pd.DataFrame(generated_data, columns=['x','y','z','time', 'vx', 'vy', 'vz', 'px', 'py', 'pz', 'distance'])
    logger = helpers.Log()
    logger.save_result(df_result)
    print(df_result.head())
    pass


if __name__ == "__main__":
    main()
