# pip install azure-storage-blob pillow

import io
import random
from statistics import mean
from azure.storage.blob import BlobServiceClient
from PIL import Image, ImageOps

# ======================
# CONSTANTS
# ======================
CONNECTION_STRING = "PASTE_CONNECTION_STRING_HERE"
CONTAINER_NAME = "PASTE_CONTAINER_NAME_HERE"

SAMPLE_SIZE = 1000
MAX_BLOBS_TO_SCAN_PER_PREFIX = 3000
MAX_SIDE_PX = 480
JPEG_QUALITY = 82

# IMPORTANT: put real prefixes/folders from your container here
PREFIXES = [
    "",              # remove this if root has millions and is too slow
    "2024/",
    "2025/",
    "2026/",
    "images/",
    "photos/",
]

ALLOWED_EXTENSIONS = (".jpg", ".jpeg")
PRINT_EVERY = 50

# ======================
# SCRIPT
# ======================

def human_size(b):
    return f"{b / (1024**2):,.2f} MiB"

def resize_to_max_side(img, max_side):
    w, h = img.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return img.copy()
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

def estimate_480p_size(data):
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    resized = resize_to_max_side(img, MAX_SIDE_PX)

    out = io.BytesIO()
    resized.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)

    return len(data), len(out.getvalue())

def main():
    service = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container = service.get_container_client(CONTAINER_NAME)

    candidates = []

    print("Collecting candidates without scanning whole container...")

    for prefix in PREFIXES:
        if len(candidates) >= SAMPLE_SIZE:
            break

        print(f"\nScanning prefix: '{prefix}'")
        scanned = 0

        for blob in container.list_blobs(name_starts_with=prefix):
            scanned += 1

            if blob.name.lower().endswith(ALLOWED_EXTENSIONS):
                candidates.append(blob.name)

            if len(candidates) >= SAMPLE_SIZE:
                break

            if scanned >= MAX_BLOBS_TO_SCAN_PER_PREFIX:
                break

        print(f"  scanned={scanned:,}, candidates_total={len(candidates):,}")

    random.shuffle(candidates)
    sample = candidates[:SAMPLE_SIZE]

    print(f"\nFinal sample size: {len(sample):,}")

    total_original = 0
    total_480p = 0
    ok = 0
    failed = 0

    for i, name in enumerate(sample, start=1):
        try:
            data = container.get_blob_client(name).download_blob().readall()
            original_size, resized_size = estimate_480p_size(data)

            total_original += original_size
            total_480p += resized_size
            ok += 1

        except Exception as e:
            failed += 1
            print(f"[WARN] Failed: {name} | {e}")

        if i % PRINT_EVERY == 0:
            saved = total_original - total_480p
            pct = saved / total_original * 100 if total_original else 0
            print(
                f"[PROGRESS] checked={i:,}/{len(sample):,}, "
                f"ok={ok:,}, failed={failed:,}, "
                f"original={human_size(total_original)}, "
                f"480p={human_size(total_480p)}, "
                f"saved={pct:.1f}%"
            )

    ratio = total_480p / total_original if total_original else 0
    saved_pct = (1 - ratio) * 100

    print("\nFINAL")
    print(f"Valid images: {ok:,}")
    print(f"Failed: {failed:,}")
    print(f"Original sample: {human_size(total_original)}")
    print(f"480p sample: {human_size(total_480p)}")
    print(f"Saved: {saved_pct:.1f}%")
    print(f"New/original ratio: {ratio:.3f}")

    container_tib = 4.38
    print(f"\nProjected for 4.38 TiB:")
    print(f"480p estimate: {container_tib * ratio:.2f} TiB")
    print(f"Saved estimate: {container_tib * (1 - ratio):.2f} TiB")

if __name__ == "__main__":
    main()