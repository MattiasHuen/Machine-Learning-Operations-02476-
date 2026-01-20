import kagglehub

import os
os.environ["KAGGLEHUB_CACHE_DIR"] = "data_cache"

# Download latest version
path = kagglehub.dataset_download("jessicali9530/lfw-dataset")

print("Path to dataset files:", path)