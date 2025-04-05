import os
import pandas as pd

# Root directory of your NAB data
DATA_ROOT = "data"

# List to store file info
file_lengths = []

# Loop through all subfolders and .csv files
for root, _, files in os.walk(DATA_ROOT):
    for file in files:
        if file.endswith(".csv"):
            filepath = os.path.join(root, file)
            try:
                df = pd.read_csv(filepath)
                file_key = os.path.relpath(filepath, DATA_ROOT).replace("\\", "/")  # Standardize path
                file_lengths.append({
                    "filename": file_key,
                    "length": len(df)
                })
            except Exception as e:
                print(f"⚠️ Failed to read {file}: {e}")

# Create DataFrame for sorting and review
length_df = pd.DataFrame(file_lengths)
length_df = length_df.sort_values("filename")

# Save to CSV
length_df.to_csv("nab_file_lengths.csv", index=False)
print("✅ File lengths saved to 'nab_file_lengths.csv'")
