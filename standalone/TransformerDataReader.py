from collections import defaultdict
import torch


class DataReader:
    def __init__(self, file_pattern):
        self.file_pattern = file_pattern

    def load_data(self, num_files):
        data_by_coords = defaultdict(list)

        for i in range(1, num_files + 1):
            filename = self.file_pattern.format(i)
            data = torch.load(filename)

            for entry in data:
                coords = tuple(entry['coordinates'])
                answer = entry['answer'].squeeze(0)  # Reshape [1, 8, 6] to [8, 6]
                data_by_coords[coords].append(answer)

        return data_by_coords
