import pandas as pd


class DataReader:
    def __init__(self, file_prefix):
        self.file_prefix = file_prefix

    def load_data(self, num_files):
        all_dfs = []

        for i in range(1, num_files + 1):
            df = pd.read_pickle(f"{self.file_prefix}{i}.pkl.zip", compression="zip")
            step = len(df) // 3360
            sampled_df = df.iloc[::step].copy()
            sampled_df = sampled_df[['x', 'y', 'z', 'time', 'latent_representation']]
            sampled_df['latent_representation'] = sampled_df['latent_representation'].apply(lambda x: x[0])
            all_dfs.append(sampled_df)

        df_combined = pd.concat(all_dfs, ignore_index=True)
        df_pivot = df_combined.pivot_table(index=['x', 'y', 'z'], columns='time', values='latent_representation',
                                           aggfunc='first')

        return df_pivot

# num_files=15
# file_prefix="/mnt/d/sources/cgan/standalone/dataset/latent_representation_for_"