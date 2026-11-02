"""
download_data.py
Downloads the MovieLens 'latest-small' dataset from GroupLens into the data/ folder.
Only needed if data/movies.csv and data/ratings.csv are missing.

Run:  python download_data.py
"""
import io
import os
import urllib.request
import zipfile

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WANTED = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading", URL, "...")
    try:
        with urllib.request.urlopen(URL, timeout=60) as resp:
            payload = resp.read()
    except Exception as err:
        print("Download failed:", err)
        print("Download the zip manually from https://grouplens.org/datasets/movielens/ "
              "and copy the CSV files from 'ml-latest-small' into the data folder.")
        return
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if base in WANTED:
                with open(os.path.join(DATA_DIR, base), "wb") as f:
                    f.write(z.read(name))
                print("saved", base)
    print("Done.")


if __name__ == "__main__":
    main()
