'''
Validation script to check for missing data across timesteps.
This file uses the same input dataframe pickled files as CleanFilesProcessor,
but focuses on validating that each x,y,z combination has data for all 1200 timesteps.
'''

from Ordered_001_Initialize import ProjectPaths
import os
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


class TimestepValidator(ProjectPaths):
    def __init__(self, filename=None):
        super().__init__(filename)
        if not hasattr(self, 'metadata_location'):
            raise AttributeError(
                "'metadata_location' is required but not set in the parent path configuration.")
        if self.metadata_location is None:
            raise ValueError("'metadata_location' is set but contains None value. A valid path must be provided.")
        self.xyz_timestep_counts = defaultdict(dict)
        self.expected_timesteps = 1200  # Expected number of timesteps

    def read_pickle_file(self, file_path):
        """Read a pickle file with different compression methods."""
        for compression in [None, 'zip', 'gzip']:
            try:
                with open(file_path, 'rb') as f:
                    if compression:
                        df = pd.read_pickle(f, compression=compression)
                    else:
                        df = pd.read_pickle(f)
                    return df
            except Exception as e:
                if compression == 'gzip':  # If we've tried all methods
                    print(f"Error reading file {file_path}: {str(e)}")
                    return None
                continue

    def validate_file(self, file_path):
        """Process a single file and count timesteps for each x,y,z combo."""
        file_name = os.path.basename(file_path)
        df = self.read_pickle_file(file_path)

        if df is None:
            print(f"Failed to read file: {file_name}")
            return None

        # Group by x,y,z and count timesteps
        counts = df.groupby(['x', 'y', 'z'])['time'].count()

        incomplete_counts = counts[counts < self.expected_timesteps]

        # Store results in our dictionary
        for xyz, count in counts.items():
            self.xyz_timestep_counts[xyz][file_name] = count

        result = {
            'file_name': file_name,
            'total_xyz_combinations': len(counts),
            'complete_data_points': (counts == self.expected_timesteps).sum(),
            'incomplete_data_points': len(incomplete_counts),
            'incomplete_details': [
                {'x': x, 'y': y, 'z': z, 'count': count, 'missing': self.expected_timesteps - count}
                for (x, y, z), count in incomplete_counts.items()
            ]
        }

        print(f"Processed {file_name}: Found {result['incomplete_data_points']} incomplete xyz combinations")
        return result

    def analyze_results(self):
        """Analyze the validation results and identify problematic x,y,z combinations."""
        # Find xyz combinations with missing timesteps across multiple files
        problematic_xyz = defaultdict(int)

        for xyz, file_counts in self.xyz_timestep_counts.items():
            incomplete_files = [f for f, count in file_counts.items() if count < self.expected_timesteps]
            if incomplete_files:
                problematic_xyz[xyz] = len(incomplete_files)

        # Sort by frequency (most problematic first)
        sorted_problematic = sorted(problematic_xyz.items(), key=lambda x: x[1], reverse=True)

        return sorted_problematic

    def export_results(self, validation_results, problematic_xyz):
        """Export the validation results to CSV files."""
        os.makedirs(os.path.join(self.output_directory, 'validation'), exist_ok=True)

        # Export file-level summary
        summary_df = pd.DataFrame(validation_results)
        summary_df.to_csv(os.path.join(self.output_directory, 'validation', 'file_summary.csv'), index=False)

        # Export problematic x,y,z coordinates
        if problematic_xyz:
            problem_df = pd.DataFrame([(x, y, z, count) for ((x, y, z), count) in problematic_xyz],
                                      columns=['x', 'y', 'z', 'incomplete_file_count'])
            problem_df.to_csv(os.path.join(self.output_directory, 'validation', 'problematic_xyz.csv'), index=False)

        # Create a detailed spreadsheet showing timestep counts for each xyz across all files
        details = []
        for (x, y, z), file_counts in self.xyz_timestep_counts.items():
            for file_name, count in file_counts.items():
                details.append({
                    'x': x,
                    'y': y,
                    'z': z,
                    'file_name': file_name,
                    'timestep_count': count,
                    'missing_timesteps': self.expected_timesteps - count if count < self.expected_timesteps else 0
                })

        if details:
            details_df = pd.DataFrame(details)
            details_df.to_csv(os.path.join(self.output_directory, 'validation', 'detailed_counts.csv'), index=False)

    def run(self):
        print(f"\nPath Configuration:")
        print(f"Input: {self.raw_input}")
        print(f"Output: {self.output_directory}")
        print(f"Metadata Location: {self.metadata_location}")
        print(f"Expected Timesteps: {self.expected_timesteps}")

        # Verify paths exist
        if not os.path.exists(self.metadata_location):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_location}")

        # Ensure output directory exists
        os.makedirs(self.output_directory, exist_ok=True)

        # Get all pickle files
        file_paths = [os.path.join(self.output_directory, file)
                      for file in os.listdir(self.output_directory)
                      if file.endswith('.pkl')]

        print(f"Found {len(file_paths)} .pkl files to validate")

        # Process files in parallel
        validation_results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(self.validate_file, file_paths))
            validation_results = [r for r in results if r is not None]

        # Analyze the validation results
        print("\nAnalyzing validation results...")
        problematic_xyz = self.analyze_results()

        # Export results to CSV
        print("Exporting results...")
        self.export_results(validation_results, problematic_xyz)

        # Print summary
        files_with_issues = sum(1 for r in validation_results if r['incomplete_data_points'] > 0)
        print(f"\nValidation Summary:")
        print(f"Total files processed: {len(validation_results)}")
        print(f"Files with incomplete data: {files_with_issues}")
        print(f"Total problematic x,y,z combinations: {len(problematic_xyz)}")

        if problematic_xyz:
            print("\nTop 10 most problematic x,y,z coordinates:")
            for i, ((x, y, z), count) in enumerate(problematic_xyz[:10], 1):
                print(f"{i}. Coordinates ({x},{y},{z}): Missing timesteps in {count} files")

        print("\nDetailed results have been saved to the output directory")


if __name__ == "__main__":
    validator = TimestepValidator()
    validator.run()
