from collections import defaultdict
import torch
import gzip
import io


class DataReader:
    def __init__(self, file_pattern):
        self.file_pattern = file_pattern

    def load_data(self, num_files):
        data_by_coords = defaultdict(list)

        for i in range(1, num_files + 1):
            filename = self.file_pattern.format(i)
            with gzip.open(filename, 'rb') as gz_file:
                buffer = io.BytesIO(gz_file.read())
            data = torch.load(buffer)
            data = data

            PERCENTAGE = 10
            core_set_size = (PERCENTAGE / 100) * len(data)
            core_set_size = int(core_set_size)
            indices = torch.linspace(0, len(data) - 1, core_set_size).long()
            data = [data[i] for i in indices]

            for entry in data:
                coords = tuple(entry['coordinates'])
                answer = entry['answer'].squeeze(0) # Reshape [1, 8, 6] to [8, 6]
                data_by_coords[coords].append(answer)

        return data_by_coords
