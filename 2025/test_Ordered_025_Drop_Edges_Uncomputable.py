import unittest
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Ordered_025_Drop_Edges_Uncomputable import ExtremeValueProcessor
import numpy as np

'''
The point of this test, was too look at the variance differences between experiments to see if that would explain
why the underlying pkl files currently are much smaller for an experiment like 3P6 as compared to 10p4, especially
given that they have the same number of rows and columns (and none of them are empty). This would seem to confirm
that for smaller variance...the compression is working much better, explaining the >25% difference in file size.
The variance between these two experiments is two orders of magnitude larger.
'''


class TestFloatColumnVariance(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.processor = ExtremeValueProcessor()
        self.output_directory = self.processor.output_directory

    def read_pickle_file(self, file_path):
        """Read a pickle file with different compression methods."""
        # Try gzip first since the error suggests it's gzip compressed
        try:
            return pd.read_pickle(file_path, compression='gzip')
        except Exception as e:
            pass

        # Then try zip
        try:
            return pd.read_pickle(file_path, compression='zip')
        except Exception as e:
            pass

        # Finally try without compression
        try:
            return pd.read_pickle(file_path)
        except Exception as e:
            print(f"Failed to read {file_path} with any compression method: {str(e)}")
            return None

    def _create_boxplots(self, all_variances, processed_files):
        """Create and save boxplots for all float columns across files."""
        # Reorganize data by column
        column_data = {}
        
        # First, find all unique columns
        all_columns = set()
        for file_stats in all_variances.values():
            all_columns.update(file_stats.keys())
        
        # For each column, collect all values from different files
        for column in all_columns:
            column_data[column] = []
            labels = []
            for filename in processed_files:
                if column in all_variances[filename]:
                    stats = all_variances[filename][column]
                    column_data[column].append(stats['variance'])
                    labels.append(filename)

        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Position for each column's box
        positions = range(len(column_data))
        
        # Create boxplots
        box_plot = ax.boxplot([data for data in column_data.values()],
                             positions=positions,
                             labels=list(column_data.keys()),
                             patch_artist=True)
        
        # Customize colors for each file
        colors = plt.cm.Set3(np.linspace(0, 1, len(processed_files)))
        
        # Rotate x-axis labels for better readability
        plt.xticks(positions, list(column_data.keys()), rotation=45, ha='right')
        
        plt.title('Distribution of Float Column Values Across Files', pad=20)
        plt.ylabel('Values')
        
        # Add grid for better readability
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=.5)
        ax.set_axisbelow(True)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save with high resolution
        plt.savefig('float_columns_analysis.png', 
                    bbox_inches='tight',
                    dpi=600)
        plt.close()

    def test_analyze_float_columns_variance(self):
        """Analyze variance in float columns and create visualization."""
        # Get all processed files
        processed_files = [f for f in os.listdir(self.output_directory)
                           if f.endswith('.pkl')]

        if not processed_files:
            self.fail("No processed files found in output directory")

        # Dictionary to store variances for each file
        all_variances = {}

        # Process each file
        for filename in processed_files:
            file_path = os.path.join(self.output_directory, filename)
            df = self.read_pickle_file(file_path)

            if df is None:
                continue

            # Get float columns
            float_columns = df.select_dtypes(include=['float32', 'float64']).columns

            # Calculate statistics for each column
            column_stats = {}
            for col in float_columns:
                stats = {
                    'variance': df[col].var(),
                    '25th': df[col].quantile(0.25),
                    '75th': df[col].quantile(0.75),
                    'median': df[col].median()
                }
                column_stats[col] = stats

            all_variances[filename] = column_stats

            # Print statistics for each file
            print(f"\nFile: {filename}")
            print("Column Statistics:")
            for col, stats in column_stats.items():
                print(f"\n{col}:")
                print(f"  Variance: {stats['variance']:.6f}")
                print(f"  25th percentile: {stats['25th']:.6f}")
                print(f"  75th percentile: {stats['75th']:.6f}")
                print(f"  Median: {stats['median']:.6f}")

        # Verify that we have data to analyze
        self.assertTrue(len(all_variances) > 0, "No data was processed")

        # Create visualizations
        self._create_boxplots(all_variances, processed_files)
        print("\nVisualization has been saved as 'float_columns_analysis.png'")


if __name__ == '__main__':
    unittest.main()