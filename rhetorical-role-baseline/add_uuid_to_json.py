import json
import uuid
import argparse

def add_uuid_to_json(input_file, output_file):
    """
    Reads a JSON file which is a list of dictionaries,
    adds a UUID to each dictionary, and writes to a new file.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file.
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}")
        return

    if not isinstance(data, list):
        print("Error: The JSON data is not a list of dictionaries.")
        return

    for item in data:
        if isinstance(item, dict):
            item['id'] = str(uuid.uuid4())

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Successfully added UUIDs to {input_file} and saved as {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add a UUID to each dictionary in a JSON file list.")
    parser.add_argument('input_file', help="The input JSON file.")
    parser.add_argument('output_file', help="The output JSON file.")
    args = parser.parse_args()

    add_uuid_to_json(args.input_file, args.output_file)
