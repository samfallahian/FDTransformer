import pandas as pd

# Replace the file path with your actual file path
file_path = "/Users/kkreth/PycharmProjects/data/DL-PTV/_combined/5p2_for_testing.csv"

try:
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)

    # Get the number of columns and their data types
    num_columns = len(df.columns)
    column_data_types = df.dtypes

    print(f"Number of columns: {num_columns}")
    print("Data types of columns:")
    print(column_data_types)

except FileNotFoundError:
    print(f"File not found at path: {file_path}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
