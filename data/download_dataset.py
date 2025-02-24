# scripts/download_dataset.py
import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, dest_path):
    """Download a file from a URL to a destination path."""
    if os.path.exists(dest_path):
        print(f"File {dest_path} already exists. Skipping download.")
        return
    print(f"Downloading {url} to {dest_path} ...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KiB
    with open(dest_path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True) as progress_bar:
        for data in response.iter_content(block_size):
            f.write(data)
            progress_bar.update(len(data))
    print("Download completed.")

def extract_zip(zip_path, extract_to):
    """Extract a zip file to a given directory if not already extracted."""
    if os.path.exists(extract_to):
        print(f"Directory {extract_to} already exists. Skipping extraction.")
        return
    print(f"Extracting {zip_path} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction completed.")

def download_dataset(data_dir="data/clevrer"):
    """
    Download and extract the CLEVRER dataset:
      - Training videos and annotations
      - Validation videos and annotations
      - Test videos
    """
    # Ensure the base data directory exists.
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Define the files to download along with their URLs, zip destination, and extraction folder.
    files_to_download = [
        {
            "url": "http://data.csail.mit.edu/clevrer/videos/train/video_train.zip",
            "zip_dest": os.path.join(data_dir, "video_train.zip"),
            "extract_to": os.path.join(data_dir, "videos", "train")
        },
        {
            "url": "http://data.csail.mit.edu/clevrer/annotations/train/annotation_train.zip",
            "zip_dest": os.path.join(data_dir, "annotation_train.zip"),
            "extract_to": os.path.join(data_dir, "annotations", "train")
        },
        {
            "url": "http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip",
            "zip_dest": os.path.join(data_dir, "video_validation.zip"),
            "extract_to": os.path.join(data_dir, "videos", "validation")
        },
        {
            "url": "http://data.csail.mit.edu/clevrer/annotations/validation/annotation_validation.zip",
            "zip_dest": os.path.join(data_dir, "annotation_validation.zip"),
            "extract_to": os.path.join(data_dir, "annotations", "validation")
        },
        {
            "url": "http://data.csail.mit.edu/clevrer/videos/test/video_test.zip",
            "zip_dest": os.path.join(data_dir, "video_test.zip"),
            "extract_to": os.path.join(data_dir, "videos", "test")
        }
    ]
    
    # Loop over each file, download and extract.
    for file in files_to_download:
        # Create the parent directory for extraction if it doesn't exist.
        extract_parent = os.path.dirname(file["extract_to"])
        if not os.path.exists(extract_parent):
            os.makedirs(extract_parent)
        
        download_file(file["url"], file["zip_dest"])
        extract_zip(file["zip_dest"], file["extract_to"])
    
    print("All files downloaded and extracted.")

if __name__ == "__main__":
    download_dataset()
