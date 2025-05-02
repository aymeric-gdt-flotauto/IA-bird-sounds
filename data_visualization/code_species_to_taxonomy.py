
import csv
import pprint
import os

dataset_path = "dataset"
csv_taxonomy_path = os.path.join(dataset_path, "eBird_Taxonomy_v2021_reduced.csv")
bird_audio_path = os.path.join(dataset_path, "bird_audio")

with open(csv_taxonomy_path, 'r') as file:
    reader = csv.reader(file)
    species_dict = {}
    row_count = 0
    for row in reader:
        if row_count == 0:
            row_count += 1
            fields = row[3:]
            continue
        species_dict[row[2]] = {field: row[i] for i, field in enumerate(fields, start=3)}
        row_count += 1

#pprint.pprint(species_dict)

# get all the folders in the bird_audio path
bird_audio_folders = os.listdir(bird_audio_path)

# match the folders name with the species_dict
for folder in bird_audio_folders:
    if folder in species_dict:
        print(f"{folder} found is {species_dict[folder]['PRIMARY_COM_NAME']}")
    else:
        print(f"{folder} not found in species_dict")



