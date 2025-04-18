from Ordered_001_Initialize import HostPreferences


class MinimalProcessor(HostPreferences):
    def __init__(self, filename="experiment.preferences"):
        super().__init__(filename)

    def run(self):
        print(f"Working with paths:")
        print(f"Input: {self.raw_input}")
        print(f"Output: {self.output_directory}")


# Example usage
if __name__ == "__main__":
    processor = MinimalProcessor()
    processor.run()
