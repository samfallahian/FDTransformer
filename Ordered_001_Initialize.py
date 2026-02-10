import socket
import ast
import logging
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HostPreferences:
    def __init__(self, filename="experiment.preferences"):
        # Resolve path relative to this file if it's not absolute and not in CWD
        if not os.path.isabs(filename) and not os.path.exists(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            potential_path = os.path.join(base_dir, filename)
            if os.path.exists(potential_path):
                filename = potential_path

        self.filename = filename
        self.hostname = self.get_hostname()
        self.root_path = None
        self.metadata_location = None
        self.logging_path = None
        self.raw_input = None
        self.output_directory = None
        self.logging_level = None
        self.training_data_path = None  # New field
        self.load_preferences()

    def get_hostname(self):
        """Get the fully qualified domain name of the current machine."""
        try:
            # Try to get the fully qualified hostname using the system's hostname command
            import subprocess
            result = subprocess.run(['hostname', '-f'], capture_output=True, text=True, check=True)
            hostname = result.stdout.strip()
        
            # If the hostname command returns an empty string, fall back to socket methods
            if not hostname:
                hostname = socket.getfqdn()
            
        except (subprocess.SubprocessError, FileNotFoundError):
            # If the hostname command fails, fall back to socket.getfqdn()
            hostname = socket.getfqdn()
        
        logger.debug(f"Current FQDN hostname: {hostname}")
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

            # Find matching configuration using regex
            matched_key = None
            for host_pattern in config.keys():
                if re.search(host_pattern, self.hostname):
                    matched_key = host_pattern
                    break

            # Verify a matching host pattern exists in the configuration
            if matched_key is None:
                available_hosts = ", ".join(config.keys())
                raise ValueError(f"No matching host pattern for '{self.hostname}'. Available patterns: {available_hosts}")

            # Get the configuration for the matched host pattern
            host_config = config[matched_key]

            # Set all configuration values (add training_data_path)
            self.root_path = host_config['root_path']
            self.metadata_location = host_config['metadata_location']
            self.logging_path = host_config['logging_path']
            self.raw_input = host_config['raw_input']
            self.output_directory = host_config['output_directory']
            self.logging_level = host_config['logging_level']
            self.training_data_path = host_config['training_data_path']  # NEW

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
        print(f"Training Data Path: {preferences.training_data_path}")  # NEW
    except Exception as e:
        print(f"Error: {e}")