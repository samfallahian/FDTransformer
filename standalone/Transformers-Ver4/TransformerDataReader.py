import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np



class DataReader:
    def __init__(self, file_prefix):
        self.file_prefix = file_prefix

    def load_data(self, num_files, start_time_frame):
        all_dfs = []

        for i in range(start_time_frame, num_files+1):
            df = pd.read_pickle(f"{self.file_prefix}{i}.pkl.zip", compression="zip")
            step = len(df) // 3360
            sampled_df = df.iloc[::step].copy()
            sampled_df = sampled_df[['x', 'y', 'z', 'time', 'latent_representation']]
            sampled_df['latent_representation'] = sampled_df['latent_representation'].apply(lambda x: x[0])
            all_dfs.append(sampled_df)

        df_combined = pd.concat(all_dfs, ignore_index=True)
        df_pivot = df_combined.pivot_table(index=['x', 'y', 'z'], columns='time', values='latent_representation',
                                           aggfunc='first')
        df_pivot_flat = df_pivot.reset_index()

        return df_pivot_flat

# data_reader = DataReader("/mnt/d/sources/cgan/standalone/dataset/latent_representation_for_")
#
# df_pivot = data_reader.load_data(15)


