import logging
import torch
from config import Config
import pandas as pd
import json

log_file = "/home/rpal_umassd_edu/logs/raw_data_process.log"
# logging.basicConfig(
#     filename=log_file,
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
logging.basicConfig(
    level=logging.INFO,filename=log_file,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class RawDataProcessor:
    def __init__(self, experiment_id: str, metadata_path: str):
        self.experiment_id = experiment_id
        self.config = Config.EXPERIMENTS[experiment_id]
        self.output_dir = self.config["output_dir"]
        # try:
        #     self.output_dir.mkdir(parents=True, exist_ok=True)
        #     logging.info(f"Output directory for experiment {self.experiment_id}: {self.output_dir}")
        # except Exception as e:
        #     logging.error(f"Error creating output directory : {e}")
        #     raise

        #Load enumerated coordinates from JSON
        try:
            with open(metadata_path, "r") as f:
                experiment_dict = json.load(f)
            self.enumerated_coords = experiment_dict.get(experiment_id, {})
            logging.info(f"Going to read Json file path...:{self.enumerated_coords}")

            if not self.enumerated_coords:
                raise ValueError(f"No enumerated coordinates found for experiment: {experiment_id}")

            # Extract enumerated x, y, z values
            self.x_enumerated = [int(x) for x in self.enumerated_coords.get("x_enumerated", [])]
            self.y_enumerated = [int(y) for y in self.enumerated_coords.get("y_enumerated", [])]
            self.z_enumerated = [int(z) for z in self.enumerated_coords.get("z_enumerated", [])]

            logging.info(f"Loaded enumerated coordinates for experiment {self.experiment_id}")
        except Exception as e:
            logging.error(f"Error loading enumerated coordinates from {metadata_path}: {e}")
            raise

    def verify_cube_formation(self, df, point):
        """Verify if a point can form a complete 5x5x5 cube using enumerated values."""
        x, y, z = point['x'], point['y'], point['z']
        logging.debug(f"Verifying cube formation for point ({x}, {y}, {z})")

        # Find valid regions
        x_start_idx = self.x_enumerated.index(x) if x in self.x_enumerated else -1
        y_start_idx = self.y_enumerated.index(y) if y in self.y_enumerated else -1
        z_start_idx = self.z_enumerated.index(z) if z in self.z_enumerated else -1

        if (
            x_start_idx == -1 or y_start_idx == -1 or z_start_idx == -1 or
            x_start_idx < 2 or y_start_idx < 2 or z_start_idx < 2 or
            x_start_idx > len(self.x_enumerated) - 3 or
            y_start_idx > len(self.y_enumerated) - 3 or
            z_start_idx > len(self.z_enumerated) - 3):
            return False

        # Generate all possible positions for a 5x5x5 cube
        x_coords = self.x_enumerated[x_start_idx - 2: x_start_idx + 3]
        y_coords = self.y_enumerated[y_start_idx - 2: y_start_idx + 3]
        z_coords = self.z_enumerated[z_start_idx - 2: z_start_idx + 3]

        # Ensure all 125 points exist in the DataFrame
        cube_points = pd.DataFrame([(x, y, z) for x in x_coords for y in y_coords for z in z_coords],
                        columns=['x', 'y', 'z'])
        matching_cube = pd.merge(cube_points, df, on=['x', 'y', 'z'], how='inner')
        return len(matching_cube) == 125

    def load_and_process_data(self):
        """Load, clean, and process data, ensuring valid cubes only."""
        logging.info(f"Processing {self.experiment_id} dataset...")

        df = pd.read_pickle(self.config["input_file"], compression="zip")
        logging.info(f"Read raw data file {self.config['input_file']} ")
        df.drop(columns=Config.COLUMNS_TO_DROP, errors="ignore", inplace=True)
        logging.info(f"Dropped columns: {Config.COLUMNS_TO_DROP}")

        tensor_data = {}
        for col, dtype in Config.COLUMN_DTYPES.items():
            if col in df.columns:
                tensor_data[col] = torch.tensor(df[col].values, dtype=dtype)
        logging.info(f"Converted columns to tensors: {list(tensor_data.keys())}")

        unique_times = df['time'].unique()
        # total_steps = len(unique_times)
        logging.info(f"Found {len(unique_times)} unique time steps to process")

        # Process each time step
        for i, time_step in enumerate(unique_times):
            time_df = df[df['time'] == time_step]
            logging.info(f"Processing time step {time_step}, rows: {len(time_df)}")

            #Filter points that are within valid ranges (excluding edge cases)
            valid_points = time_df[
                (time_df['x'].isin(self.x_enumerated[2:-2])) &
                (time_df['y'].isin(self.y_enumerated[2:-2])) &
                (time_df['z'].isin(self.z_enumerated[2:-2]))]
            logging.info(f"Valid points after edge filtering: {len(valid_points)} for time {time_step}")

            # Verify cubes and log invalid cases
            valid_data = []
            for _, point in valid_points.iterrows():
                if self.verify_cube_formation(time_df, point):
                    valid_data.append(point)
                else:
                    logging.debug(f"Point {point['x']}, {point['y']}, {point['z']} failed cube formation verification")

            # Save valid data
            if valid_data:
                output_df = pd.DataFrame(valid_data)
                print(output_df.head())
                # output_file = self.output_dir / f"{int(time_step)}.pkl"
                # output_df.to_pickle(output_file, compression="zip")
                # logging.info(f"Saved {len(valid_data)} valid points to {output_file}")

            # Periodic updates for large datasets
            # if (i + 1) % 100 == 0:
            #     logging.info(f"Processed {i + 1}/{len(unique_times)} timesteps")

            # for idx, time_step in enumerate(unique_times, 1):
        #     time_mask = tensor_data['time'] == time_step
        #
        #     time_data = {
        #         col: tensor_data[col][time_mask]
        #         for col in tensor_data.keys()
        #     }
            # output_file = self.output_dir / f"{idx}.pkl"

if __name__ == "__main__":
    metadata_path = "/home/rpal_umassd_edu/cgan_deployment/configs/Umass_experiments.txt"
    logging.info(f"Going to read Json file path...")

    for experiment_id in Config.EXPERIMENTS.keys():
        try:
            data_processor = RawDataProcessor(experiment_id, metadata_path)
            data_processor.load_and_process_data()
            logging.info(f"Completed processing for experiment {experiment_id}")
        except Exception as e:
            logging.error(f"Error processing experiment {experiment_id}: {e}")