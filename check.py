import nltk
import shutil
import os

# Print the NLTK data path
print(nltk.data.path)

# Remove and recreate the nltk_data directory
for path in nltk.data.path:
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            os.makedirs(path)
            print(f"Cleared NLTK data at {path}")
        except Exception as e:
            print(f"Could not clear {path}: {e}")