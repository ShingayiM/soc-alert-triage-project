"""Download the NSL-KDD train/test CSVs into ./data (run from the project root)."""
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/master/"
FILES = ["NSL_KDD_Train.csv", "NSL_KDD_Test.csv"]

os.makedirs("data", exist_ok=True)
for name in FILES:
    dest = os.path.join("data", name)
    if os.path.exists(dest):
        print(f"{dest} already exists, skipping")
        continue
    print(f"Downloading {name} ...")
    urllib.request.urlretrieve(BASE + name, dest)
print("Done.")
