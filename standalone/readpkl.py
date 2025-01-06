import torch
from config import Config
import pandas as pd
import pickle
from pathlib import Path


class RawDataProcessor:
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        print(self.experiment_id)
        self.output_dir = Path("/work/pi_bseyedaghazadeh_umassd_edu/DL-PTV/roshni/test_dir")
        self.input_dir=Path("/work/pi_bseyedaghazadeh_umassd_edu/DL-PTV/roshni/3p6/303.pkl")
        self.config = Config.EXPERIMENTS[experiment_id]
        print(self.output_dir,self.input_dir)

    def load_and_process_data(self):
        print(f"Processing {self.experiment_id} dataset...")
        df = pd.read_pickle(self.input_dir, compression="zip")
        df_labels = df[(df["time"] == 303) & (df["distance"] == 150)]
        print(df_labels)
        output_file = self.output_dir / f"demo.pkl"
        df_labels.to_pickle(output_file, compression="zip")
        # torch.save(df_labels,output_file)
        print(f"File Written Successfully")

if __name__ == "__main__":
    for experiment_id in Config.EXPERIMENTS.keys():
        try:
            data_processor = RawDataProcessor(experiment_id)
            data_processor.load_and_process_data()
        except Exception as e:
            print(f"Error processing experiment {experiment_id}: {e}")
