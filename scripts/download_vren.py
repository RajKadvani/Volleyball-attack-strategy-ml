"""
Download script for the official VREN Volleyball Rally Dataset.
Source: https://github.com/haotianxia/VREN
"""

import os
import sys
import urllib.request
import urllib.error

DATA_URLS = [
    "https://raw.githubusercontent.com/haotianxia/VREN/main/data/dataset_full.csv",
    "https://raw.githubusercontent.com/haotianxia/VREN/main/dataset_full.csv"
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "vren_full_dataset.csv")

def download_vren():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Attempting to download VREN dataset from GitHub...")
    success = False

    for url in DATA_URLS:
        try:
            print(f"Connecting to: {url}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(OUTPUT_FILE, 'wb') as out_file:
                total_bytes = 0
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    total_bytes += len(chunk)
            print(f"Successfully downloaded {total_bytes:,} bytes to {OUTPUT_FILE}")
            success = True
            break
        except Exception as e:
            print(f"Download failed from {url}: {e}")

    if not success:
        print("\nNote: Network download failed or unavailable.")
        print("You can still use the local benchmark dataset at 'data/sample_vren_dataset.csv'")
        print("or run 'python scripts/generate_sample_data.py' to generate realistic benchmark data.")

if __name__ == "__main__":
    download_vren()
