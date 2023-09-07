import json

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def create_json_object(driver_source, model_source, std_out):
    data = {
        "tasks": [
            {
                "driver_source": driver_source,
                "model_source": model_source,
                "std_out": std_out,
                "query": "Assist with the NoneType error please."
            }
        ]
    }
    return json.dumps(data)

if __name__ == "__main__":
    # Read text from files
    driver_source = read_file("/Users/kkreth/PycharmProjects/cgan/standalone/train_conv.py")
    model_source = read_file("/standalone/HybrdidAutoencoder.py")
    std_out = read_file("/Users/kkreth/PycharmProjects/cgan/standalone/console_output/console.txt")

    # Create JSON object
    json_obj = create_json_object(driver_source, model_source, std_out)

    print(json_obj)