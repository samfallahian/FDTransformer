import pickle

# Path to your pickled dataframe
pickle_file_path = '/Users/kkreth/PycharmProjects/data/DL-PTV/combined_data_for_training_AE.dataframe.pkl'

# Load the pickled DataFrame
with open(pickle_file_path, 'rb') as file:
    df_pandas = pickle.load(file)

# Print the first 5 rows
print(df_pandas.head())