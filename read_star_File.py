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

    print("=" * 60)
    print(f"⭐ STAR DATA: {star_name}")
    print("=" * 60)
    print(f"📂 File : {file_path}")
    print(f"📦 Arrays: {', '.join(data.files)}")
    print("=" * 60)

    for key in data.files:
        arr = data[key]

        print(f"\n🔹 {key.upper()}")
        print("-" * 60)
        print(f"Shape : {arr.shape}")
        print(f"Dtype : {arr.dtype}")

        # Show preview
        if arr.size > 10:
            print("Values:")
            print(" ", arr[:5], "...", arr[-5:])
        else:
            print("Values:")
            print(" ", arr)

    print("\n" + "=" * 60)
    print("✅ Data loaded successfully.")
    print("=" * 60)

    data.close()

else:
    print("=" * 60)
    print(f"❌ No data found for star '{star_name}'.")
    print("=" * 60)