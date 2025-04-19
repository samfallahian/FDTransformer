import socket
import ast


class HostPreferences:
    def __init__(self, filename="experiment.preferences"):
        self.filename = filename
        self.hostname = self.get_hostname()
        self.root_path = None
        self.metadata_location = None
        self.logging_path = None
        self.raw_input = None
        self.output_directory = None
        self.logging_level = None
        self.load_preferences()

    def get_hostname(self):
        """Get the hostname of the current machine."""
        hostname = socket.gethostname()
        print(f"Current hostname: {hostname}")  # Debug line
        return hostname.lower()  # Convert to lowercase for case-insensitive matching

    def load_preferences(self):
        """Load preferences for the current host from the configuration file."""
        try:
            # Read and parse the configuration file
            with open(self.filename, 'r') as file:
                content = file.read()

            # Parse the dictionary
            config = ast.literal_eval(content)

            # Convert all host keys to lowercase for case-insensitive matching
            config = {k.lower(): v for k, v in config.items()}

            # Verify the current host exists in the configuration
            if self.hostname not in config:
                available_hosts = ", ".join(config.keys())
                raise ValueError(f"Host '{self.hostname}' not found. Available hosts: {available_hosts}")

            # Get the configuration for the current host
            host_config = config[self.hostname]

            # Set all configuration values
            self.root_path = host_config['root_path']
            self.metadata_location = host_config['metadata_location']
            self.logging_path = host_config['logging_path']
            self.raw_input = host_config['raw_input']
            self.output_directory = host_config['output_directory']
            self.logging_level = host_config['logging_level']

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file '{self.filename}' not found.")
        except KeyError as e:
            raise ValueError(f"Missing required configuration field: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading preferences: {str(e)}")


# Example usage:
if __name__ == "__main__":
    try:
        preferences = HostPreferences()
        print("\nConfiguration loaded successfully:")
        print(f"Host: {preferences.hostname}")
        print(f"Metadata Location: {preferences.metadata_location}")
        print(f"Root Path: {preferences.root_path}")
        print(f"Logging Path: {preferences.logging_path}")
        print(f"Raw Input: {preferences.raw_input}")
        print(f"Output Directory: {preferences.output_directory}")
        print(f"Logging Level: {preferences.logging_level}")
    except Exception as e:
        print(f"Error: {e}")