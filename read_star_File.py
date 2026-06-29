import os
import numpy as np

# Folder containing the .npz files
DATA_FOLDER = "./data/raw"

# Get star name from user
star_name = input("Enter Star Name: ").strip()

# Construct file path
file_path = os.path.join(DATA_FOLDER, f"{star_name}.npz")

# Check if file exists
if os.path.exists(file_path):
    data = np.load(file_path)

    print(f"\nLoaded: {file_path}")
    print("Available arrays:", data.files)

    # Print all arrays
    for key in data.files:
        print(f"\n--- {key} ---")
        print(data[key])

    data.close()
else:
    print(f"No data found for star '{star_name}'.")