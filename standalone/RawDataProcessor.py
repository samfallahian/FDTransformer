import argparse
import logging
import pandas as pd
import json
from itertools import product
from typing import List
from config import Config
from TransformLatent import FloatConverter

log_file = Config.LOG_FILE_RAW_DATA_PROCESSOR
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def locate_coordinates(value: int, enumerated_list: List[int]) -> List[int]:
    if value not in enumerated_list:
        logging.warning(f"Value {value} not found in enumerated list.")
        raise ValueError(f"Value {value} not found in enumerated list.")
    index = enumerated_list.index(value)

    logging.info(f"Index: {index}")
    if index < 2 or index >= len(enumerated_list) - 2:
        logging.warning(f"Coordinate {value} out of range for cube formation.")
        raise ValueError("Coordinate out of range for cube formation.")
    return enumerated_list[index - 2:index + 3]

class CoordinatesAnalyser:
    def __init__(self, x_enumerated: List[int], y_enumerated: List[int], z_enumerated: List[int]):
        self.x_enumerated = x_enumerated
        self.y_enumerated = y_enumerated
        self.z_enumerated = z_enumerated

    def analyze(self, df, x, y, z) -> bool:
        try:
            x_values = locate_coordinates(x, self.x_enumerated)
            y_values = locate_coordinates(y, self.y_enumerated)
            z_values = locate_coordinates(z, self.z_enumerated)

            # Generate all combinations of x, y, z values
            all_combinations = list(product(x_values, y_values, z_values))
            logging.info(f"Generated {len(all_combinations)} combinations for ({x}, {y}, {z})")

            if len(all_combinations) != 125:
                logging.error("Cube combinations are incomplete. Possible edge case.")
                raise ValueError("Coordinates not compatible for 5x5x5 cube.")

            # Create a DataFrame from all combinations
            cube_df = pd.DataFrame(all_combinations, columns=['x', 'y', 'z'])

            # Check which of these combinations exist in the provided DataFrame
            matching_cube = pd.merge(cube_df, df, on=['x', 'y', 'z'], how='inner')
            logging.info(f"Matching points found: {len(matching_cube)}")
            return len(matching_cube) == 125
        except Exception as e:
            logging.error(f"Error analyzing cube for ({x}, {y}, {z}): {e}")
            return False

class RawDataProcessor:
    def __init__(self, experiment_id: str):
        self.analyzer = None
        self.z_enumerated = None
        self.y_enumerated = None
        self.x_enumerated = None
        self.config = Config.EXPERIMENTS[experiment_id]
        self.metadata_path = Config.METADATA_PATH
        self.output_dir = self.config["output_dir"]
        self.converter = FloatConverter()
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Output directory for experiment {experiment_id}: {self.output_dir}")
        except Exception as e:
            logging.error(f"Error creating output directory: {e}")
            raise
        self.load_enumerated_coords(experiment_id)

    def load_enumerated_coords(self, experiment_id: str):
        try:
            with open(self.metadata_path, "r") as f:
                experiment_dict = json.load(f)
            enumerated_coords = experiment_dict.get(experiment_id, {})
            if not enumerated_coords:
                raise ValueError(f"No enumerated coordinates found for experiment: {experiment_id}")
            self.x_enumerated = enumerated_coords.get("x_enumerated", [])
            self.y_enumerated = enumerated_coords.get("y_enumerated", [])
            self.z_enumerated = enumerated_coords.get("z_enumerated", [])
            self.analyzer = CoordinatesAnalyser(self.x_enumerated, self.y_enumerated, self.z_enumerated)
        except Exception as e:
            logging.error(f"Error loading enumerated coordinates: {e}")
            raise

    def process_timestep(self, df: pd.DataFrame, time_step: int):
        time_df = df[df['time'] == time_step]
        logging.info(f"Processing timestep {time_step} with {len(time_df)} rows.")
        logging.info(f"Single row from time_df: {time_df.iloc[0].to_dict()}")

        valid_points = time_df[
            (time_df['x'].isin(self.x_enumerated[2:-2])) &
            (time_df['y'].isin(self.y_enumerated[2:-2])) &
            (time_df['z'].isin(self.z_enumerated[2:-2]))
        ]
        logging.info(f"valid_points length:  {len(valid_points)}")
        valid_data = []
        coordinates_analyser = CoordinatesAnalyser(self.x_enumerated, self.y_enumerated, self.z_enumerated)
        for point in valid_points.itertuples(index=False):
            logging.info(f"Point: {point.x, point.y, point.z, point.vx, point.vy, point.vz, point.time, point.distance}")
            if coordinates_analyser.analyze(time_df, point.x, point.y, point.z):
                valid_data.append([point.x, point.y, point.z, point.vx, point.vy, point.vz, point.time, point.distance])

        if valid_data:
            valid_data_df = pd.DataFrame(valid_data, columns=['x','y','z','vx', 'vy', 'vz','time','distance'])
            logging.info(f"Dataframe converted")

            output_df = self.convert_velocities(valid_data_df)
            # Save output
            output_file = self.output_dir / f"{time_step}.pkl"
            output_df.to_pickle(output_file, compression="zip")
            logging.info(f"Saved {len(output_df)} valid points to {output_file}")
        else:
            logging.warning(f"No valid data found for timestep {time_step}")

    def convert_velocities(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.debug(f"Initial velocities: {df[['vx', 'vy', 'vz']].head().to_dict()}")
        try:
            df[['vx', 'vy', 'vz']] = df[['vx', 'vy', 'vz']].map(self.converter.convert).astype('float32')
            return df
        except ValueError as e:
            logging.error(f"Error converting velocities: {e}")
            raise

    def load_and_process_data(self):
        df = pd.read_pickle(self.config["input_file"], compression="zip")
        logging.info(f"Loaded raw data: {self.config['input_file']} with {len(df)} rows.")

        if Config.COLUMNS_TO_DROP:
            df = df.drop(columns=Config.COLUMNS_TO_DROP, errors='ignore')
            logging.info(f"Dropped columns: {Config.COLUMNS_TO_DROP}")

        for col, dtype in Config.COLUMN_DTYPES.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        logging.info(f"Updated column dtypes: {df.dtypes.to_dict()}")

        unique_times = df['time'].unique()
        logging.info(f"Found {len(unique_times)} time_step: {unique_times}")
        logging.info(f"Found {unique_times} unique timesteps: {unique_times}")

        for idx, time_step in enumerate(unique_times, 1):
            self.process_timestep(df, time_step)

            # Periodic updates for large datasets
            if idx % 100 == 0:
                logging.info(f"Processed {idx}/{len(unique_times)} timesetps")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw data for X,Y,Z coordinates")
    parser.add_argument("--experiment_id", type=str, required=True, help="Experiment ID to process")
    args = parser.parse_args()
    experiment_id = args.experiment_id
    for experiment_id in Config.EXPERIMENTS.keys():
        try:
            processor = RawDataProcessor(experiment_id)
            processor.load_and_process_data()
            logging.info(f"Completed processing for experiment {experiment_id}")
        except Exception as e:
            logging.error(f"Error processing experiment {experiment_id}: {e}")
