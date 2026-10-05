import os
import sys
import random
import data_prep
import dataset_split

sys.stdout.reconfigure(encoding="utf-8")

input_folder = "input_folder"
output_folder = "output_folder"
random.seed(111)

data_prep.download_dataset(input_folder)
# data_prep.create_subset_dataset(input_folder)
data_prep.remove_corrupted_files(input_folder)
data_prep.remove_duplicate_files(input_folder)

for root, dirs, files in os.walk(input_folder):
    rel_path = os.path.relpath(root, input_folder)
    if rel_path == '.':
        continue

    valid_files = [ os.path.join(root, f) for f in files if f.lower().endswith(data_prep.VALID_EXTENSIONS) ]
    random.shuffle(valid_files)
    target_files = dataset_split.split_files(valid_files)

    for split in target_files.keys():
        target_dir = os.path.join(output_folder, split, rel_path)
        os.makedirs(target_dir, exist_ok=True)
        dataset_split.save_files_to_dir(target_files[split], target_dir, split)

