import unittest
import os
import pandas as pd
from Ordered_020_NetNewCleanFiles import CleanFilesProcessor


class TestCleanFilesProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.processor = CleanFilesProcessor()

    def test_input_output_row_counts_match(self):
        """Test that all input and output files have matching row counts."""
        # Get all input files
        input_files = [f for f in os.listdir(self.processor.raw_input) if f.endswith('.pkl')]

        # Dictionary to store row counts
        input_counts = {}
        output_counts = {}

        # Process input files
        print("\nRow counts for input files:")
        for input_file in input_files:
            input_path = os.path.join(self.processor.raw_input, input_file)
            input_df = self.processor.read_pickle_file(input_path)
            row_count = len(input_df)
            input_counts[input_file] = row_count
            print(f"{input_file}: {row_count} rows")

        # Process output files
        print("\nRow counts for output files:")
        for input_file in input_files:
            output_path = os.path.join(self.processor.output_directory, input_file)
            output_df = pd.read_pickle(output_path, compression='gzip')
            row_count = len(output_df)
            output_counts[input_file] = row_count
            print(f"{input_file}: {row_count} rows")

        # Assert that row counts match for each file pair
        for filename in input_files:
            self.assertEqual(
                input_counts[filename],
                output_counts[filename],
                f"Row count mismatch for {filename}: "
                f"Input has {input_counts[filename]} rows, "
                f"Output has {output_counts[filename]} rows"
            )

        # Assert that we actually processed 11 files
        self.assertEqual(len(input_files), 11,
                         f"Expected 11 files but found {len(input_files)}")


if __name__ == '__main__':
    unittest.main()