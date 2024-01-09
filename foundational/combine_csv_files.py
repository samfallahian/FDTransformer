import os
import pandas as pd

# Set the directory where your CSV files are located
directory_path = '/Users/kkreth/PycharmProjects/data/DL-PTV/_combined'

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# Initialize a list to store the data
data_frames = []

# Loop through each CSV file and append its contents to the data list
for csv_file in csv_files:
    file_path = os.path.join(directory_path, csv_file)
    df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
    data_frames.append(df)

# Combine all data frames into one
combined_data = pd.concat(data_frames, ignore_index=True)

# Save the combined data to a new CSV file
combined_data.to_csv('combined_data.csv', index=False)

print("CSV files have been successfully combined and saved as 'combined_data.csv'")