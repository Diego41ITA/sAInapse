import zipfile
import requests
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def downlaod_dataset():
# URL of the zip file
    url = "https://ninapro.hevs.ch/files/DB5_Preproc/s1.zip"  # Example for s1

    # Specify the folder where you want to download the zip file
    download_folder = "/app/cat/data/datasets"  # Change this to your desired folder
    os.makedirs(download_folder, exist_ok=True)  # Make sure the folder exists


    zip_filename = "s1.zip"

    # Download the file if it doesn't already exist
    if not os.path.exists(zip_filename):
        print(f"Downloading {url} to {zip_filename} ...")
        r = requests.get(url, allow_redirects=True)
        with open(zip_filename, 'wb') as f:
            f.write(r.content)
        print("Download complete.")
    else:
        print(f"{zip_filename} already present, skipping download.")

    # Specify the folder where you want to extract the contents
    extract_folder = "/app/cat/data/datasets"  
    os.makedirs(extract_folder, exist_ok=True)

    # Unzip the file
    with zipfile.ZipFile(zip_filename, 'r') as zf:
        zf.extractall(extract_folder)

    print(f"Files extracted to {extract_folder}")


if __name__ == "__main__":
    downlaod_dataset()
