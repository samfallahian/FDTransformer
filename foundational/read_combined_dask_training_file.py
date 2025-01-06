import pandas as pd
import pickle

# Define the path of the pickled file
pickle_file = '/Users/kkreth/PycharmProjects/data/DL-PTV/combined_data_for_training_AE_df.pkl'  # Replace with your file path

# Load the DataFrame from the pickled file
with open(pickle_file, 'rb') as file:
    df = pickle.load(file)


# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# Print 5 random rows from the DataFrame
print(df.sample(5))


