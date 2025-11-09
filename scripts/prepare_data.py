"""Prepare dataset:
- Optionally convert DICOM to PNG
- Normalize to [0,1] later in tf pipeline
- Build CSV split files if needed (not used by default; we leverage directory splits)
"""
import os, glob, shutil
import pydicom
from PIL import Image
import numpy as np

def dicom_to_png(dicom_path, out_path):
    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array.astype(np.float32)
    p1, p99 = np.percentile(arr, (1, 99))
    arr = np.clip((arr - p1) / (p99 - p1 + 1e-6), 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)

if __name__ == "__main__":
    print("This script provides helpers; adapt as needed for your dataset layout.")
