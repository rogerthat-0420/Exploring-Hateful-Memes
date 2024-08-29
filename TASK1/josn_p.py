import jsonlines

# Specify the path to your train.jsonl file
jsonl_file_path = "./hateful_memes/train.jsonl"

# Initialize an empty list to store labels
labels = []

# Read the jsonl file and extract labels
with jsonlines.open(jsonl_file_path) as reader:
    for entry in reader:
        # Extract the label field and append to the list
        labels.append(entry["label"])

# Display the extracted labels
print(labels)

