import zipfile
import os

# Correct path to the zip file
zip_path = "unzipped_dataset/CK+48.zip"
extract_to = "data"

# Make sure the target folder exists
os.makedirs("data", exist_ok=True)

# Unzip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("✅ Dataset successfully unzipped to:", extract_to)