from collections import defaultdict
import torch
import gzip
import io


class DataReader:
    def __init__(self, file_pattern):
        self.file_pattern = file_pattern

    def load_data(self, num_files, data_chunk_percentage, data_chunk):
        data_by_coords = defaultdict(list)

        for i in range(1, num_files + 1):
            filename = self.file_pattern.format(i)
            with gzip.open(filename, 'rb') as gz_file:
                buffer = io.BytesIO(gz_file.read())
            data = torch.load(buffer)

            core_set_size = int((data_chunk_percentage / 100) * len(data))

            # print("len core: ", len(data[data_chunk * core_set_size:(data_chunk + 1) * core_set_size]))
            # print("len core: ", len(data[2*core_set_size:3*core_set_size]))
            data = data[data_chunk * core_set_size:(data_chunk + 1) * core_set_size]

            for entry in data:
                coords = tuple(entry['coordinates'])
                answer = entry['answer'].squeeze(0)  # Reshape [1, 8, 6] to [8, 6]
                data_by_coords[coords].append(answer)

        return data_by_coords
