# CarbonBloom Pro

CarbonBloom Pro is a Streamlit application for urban heat and vegetation analysis. Upload satellite imagery to detect hard-surface / heat-risk areas using a segmentation model, visualize overlays, get engineering recommendations, and download results.

## Key Files
- `03_app.py` — Streamlit UI and inference pipeline.
- `carbon_bloom_model.pth` — PyTorch model weights (place your trained weights here).
- `dataset/` — (optional) images and masks. Consider using Git LFS for large files.

## Requirements
- Python 3.8–3.11
- pip
- Optional: GPU + CUDA for faster inference

## Install
1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install streamlit pillow numpy opencv-python torch torchvision segmentation-models-pytorch
```

- If you need CUDA-enabled PyTorch, follow the official PyTorch install instructions to get the right wheel for your GPU.

## Configuration
- Place your trained model at `carbon_bloom_model.pth` in the project root.
- Store images under `dataset/images/` if desired.

## Run the App

```bash
python -m streamlit run "03_app.py"
```

Open the URL printed by Streamlit (usually http://localhost:8501).

## How It Works
- Loads a U-Net segmentation model (ResNet18 encoder).
- Uploaded images are preprocessed and passed through the model.
- Thresholding uses the `AI Detection Power` slider, and `Nature Protection` shields green areas.
- The app overlays detections, computes simple metrics (density, estimated surface temperature, CO₂ offset), and stores session history.

If push fails due to large files, move offending files to LFS or host externally.

## Alternative: External Hosting
If dataset is very large, upload to S3 / Drive / Zenodo and provide a `01_download_data.py` script.

Example:
```python
# 01_download_data.py
import os
import requests

URL = "https://example.com/path/to/dataset.zip"
OUT = "dataset/dataset.zip"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with requests.get(URL, stream=True) as r:
    r.raise_for_status()
    with open(OUT, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
print("Downloaded to", OUT)
```

## Troubleshooting
- `streamlit` not found: ensure virtual env active and `streamlit` installed, run `python -m streamlit run "03_app.py"`.
- Model load errors: verify `carbon_bloom_model.pth` and device mapping.
- Slow inference: use GPU build of PyTorch and ensure CUDA drivers are installed.
- Large file push error: use `git lfs track` or host files externally.

## Security & Privacy
Do not commit sensitive or PII-containing imagery to public repos. Use anonymized datasets or private cloud storage.

---
