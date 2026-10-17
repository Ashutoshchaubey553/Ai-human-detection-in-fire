"""
download_models.py
Ensures model weights and configuration files exist and are not Git LFS pointer files.
Used during build (e.g. Render buildCommand) before starting the Flask app.
"""

import os
import re
import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Google Drive File IDs
GDRIVE_FILES = {
    "yolov3.weights": {
        "id": "1N13RalYAddLPKnXILhSMVojXL8OtUC_X",
        "min_size": 100 * 1024 * 1024,  # ~248 MB
        "fallback_url": "https://pjreddie.com/media/files/yolov3.weights",
    },
    "yolov3.cfg": {
        "id": "10GxrblhK-KwoavRD6vANYsT0y_CUs2dp",
        "min_size": 5 * 1024,  # ~8.3 KB
        "fallback_url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    },
    "coco.names": {
        "id": "1lkM4bD9lGDPm4BolBhODPXnQncJulVr_",
        "min_size": 200,  # ~625 B
        "fallback_url": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
    },
}


def is_lfs_pointer_or_corrupt(file_path: Path, min_size: int) -> bool:
    """Check if file is missing, smaller than expected, or an unpulled Git LFS pointer."""
    if not file_path.exists():
        return True
    try:
        size = file_path.stat().st_size
        if size < min_size:
            return True
        with open(file_path, "rb") as f:
            header = f.read(50)
            if b"version https://git-lfs" in header:
                return True
    except Exception as e:
        print(f"[!] Error checking {file_path.name}: {e}")
        return True
    return False


def download_from_gdrive(file_id: str, dest_path: Path) -> bool:
    """Download file from Google Drive, handling direct download and virus scan confirmation."""
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "")
            # Direct binary download
            if "text/html" not in content_type:
                with open(tmp_path, "wb") as f:
                    while chunk := resp.read(1024 * 1024):
                        f.write(chunk)
                if tmp_path.exists():
                    tmp_path.replace(dest_path)
                return True

            html = resp.read().decode("utf-8", errors="ignore")

        # Large file confirmation page: extract uuid & confirm parameter
        uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', html)
        confirm_match = re.search(r'name="confirm"\s+value="([^"]+)"', html)
        uuid = uuid_match.group(1) if uuid_match else None
        confirm = confirm_match.group(1) if confirm_match else "t"

        if uuid:
            dl_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm={confirm}&uuid={uuid}"
        else:
            token_match = re.search(r"confirm=([0-9A-Za-z_]+)", html)
            token = token_match.group(1) if token_match else "t"
            dl_url = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"

        req2 = urllib.request.Request(dl_url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req2, timeout=600) as resp2, open(tmp_path, "wb") as f:
            total_read = 0
            while chunk := resp2.read(2 * 1024 * 1024):
                f.write(chunk)
                total_read += len(chunk)
                if total_read % (50 * 1024 * 1024) < len(chunk):
                    print(f"    Progress: {total_read // (1024 * 1024)} MB downloaded...")

        if tmp_path.exists():
            tmp_path.replace(dest_path)
        return True

    except Exception as e:
        print(f"    [!] Google Drive download failed: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def download_from_url(url: str, dest_path: Path) -> bool:
    """Fallback download from a direct URL."""
    print(f"    Attempting fallback download from: {url}")
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    try:
        with urllib.request.urlopen(req, timeout=600) as resp, open(tmp_path, "wb") as f:
            total_read = 0
            while chunk := resp.read(2 * 1024 * 1024):
                f.write(chunk)
                total_read += len(chunk)
                if total_read % (50 * 1024 * 1024) < len(chunk):
                    print(f"    Progress: {total_read // (1024 * 1024)} MB downloaded...")

        if tmp_path.exists():
            tmp_path.replace(dest_path)
        return True
    except Exception as e:
        print(f"    [!] Fallback download failed: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def main():
    print("=== Checking model files ===")
    all_ok = True

    # 1. Check fire_detection_model.h5
    fire_model = MODELS_DIR / "fire_detection_model.h5"
    if is_lfs_pointer_or_corrupt(fire_model, min_size=1 * 1024 * 1024):
        print(f"[!] Warning: {fire_model.name} appears missing or corrupt ({fire_model.stat().st_size if fire_model.exists() else 0} bytes).")
        all_ok = False
    else:
        print(f"[OK] {fire_model.name} is present ({fire_model.stat().st_size // (1024 * 1024)} MB).")

    # 2. Check YOLO files
    for filename, info in GDRIVE_FILES.items():
        target = MODELS_DIR / filename
        if is_lfs_pointer_or_corrupt(target, info["min_size"]):
            print(f"[*] {filename} is missing or an unpulled Git LFS pointer. Downloading...")
            success = download_from_gdrive(info["id"], target)
            if not success and info.get("fallback_url"):
                success = download_from_url(info["fallback_url"], target)

            if not success or is_lfs_pointer_or_corrupt(target, info["min_size"]):
                print(f"[ERROR] Failed to obtain valid {filename}.")
                all_ok = False
            else:
                print(f"[OK] Successfully downloaded {filename} ({target.stat().st_size} bytes).")
        else:
            print(f"[OK] {filename} is already present ({target.stat().st_size} bytes).")

    if not all_ok:
        sys.exit(1)

    print("=== All model files verified successfully! ===")


if __name__ == "__main__":
    main()
