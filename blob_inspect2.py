# pip install azure-storage-blob pillow

import random
import io
from statistics import mean
from azure.storage.blob import BlobServiceClient
from PIL import Image, ImageOps

# ======================
# CONSTANTS
# ======================
CONNECTION_STRING = "PASTE_CONNECTION_STRING_HERE"
CONTAINER_NAME = "PASTE_CONTAINER_NAME_HERE"

SAMPLE_SIZE = 1000
RANDOM_SEED = 42

LIST_PROGRESS_EVERY = 50_000
DOWNLOAD_PROGRESS_EVERY = 50

JPEG_QUALITY = 82
MAX_SIDE_PX = 480

ALLOWED_EXTENSIONS = (".jpg", ".jpeg")

# ======================
# SCRIPT
# ======================

def human_size(num_bytes: int) -> str:
    return (
        f"{num_bytes:,} bytes | "
        f"{num_bytes / (1024 ** 2):,.2f} MiB | "
        f"{num_bytes / (1024 ** 3):,.2f} GiB"
    )


def resize_to_max_side(img: Image.Image, max_side: int) -> Image.Image:
    width, height = img.size
    current_max = max(width, height)

    if current_max <= max_side:
        return img.copy()

    scale = max_side / current_max
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def reservoir_sample_blobs(container_client):
    random.seed(RANDOM_SEED)

    sample = []
    total_seen = 0
    jpeg_candidates = 0

    print("Listing blobs and building random JPEG sample...")

    for blob in container_client.list_blobs():
        total_seen += 1
        name = blob.name

        if not name.lower().endswith(ALLOWED_EXTENSIONS):
            continue

        jpeg_candidates += 1

        item = {
            "name": name,
            "size": blob.size or 0,
        }

        if len(sample) < SAMPLE_SIZE:
            sample.append(item)
        else:
            j = random.randint(0, jpeg_candidates - 1)
            if j < SAMPLE_SIZE:
                sample[j] = item

        if total_seen % LIST_PROGRESS_EVERY == 0:
            print(
                f"[LIST PROGRESS] total_seen={total_seen:,}, "
                f"jpeg_candidates={jpeg_candidates:,}, "
                f"sample_size={len(sample):,}"
            )

    print(
        f"Listing done. total_seen={total_seen:,}, "
        f"jpeg_candidates={jpeg_candidates:,}, "
        f"sample_size={len(sample):,}"
    )

    return sample, total_seen, jpeg_candidates


def estimate_480p_size(blob_bytes: bytes):
    original_size = len(blob_bytes)

    try:
        img = Image.open(io.BytesIO(blob_bytes))
        img = ImageOps.exif_transpose(img)
    except Exception:
        return None

    original_width, original_height = img.size

    if img.mode != "RGB":
        img = img.convert("RGB")

    resized = resize_to_max_side(img, MAX_SIDE_PX)
    resized_width, resized_height = resized.size

    output = io.BytesIO()
    resized.save(
        output,
        format="JPEG",
        quality=JPEG_QUALITY,
        optimize=True,
        progressive=True,
    )

    resized_size = len(output.getvalue())

    return {
        "original_size": original_size,
        "resized_size": resized_size,
        "original_width": original_width,
        "original_height": original_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "was_downscaled": resized_size < original_size,
    }


def main():
    service = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container = service.get_container_client(CONTAINER_NAME)

    sample, total_seen, jpeg_candidates = reservoir_sample_blobs(container)

    checked = 0
    valid_images = 0
    failed = 0
    downscaled_count = 0

    total_original = 0
    total_480p = 0
    ratios = []

    print("\nDownloading sampled JPEGs and estimating 480p size...")
    print("-" * 80)

    for item in sample:
        checked += 1
        name = item["name"]

        try:
            data = container.get_blob_client(name).download_blob().readall()
            result = estimate_480p_size(data)

            if result is None:
                failed += 1
                continue

            valid_images += 1

            original_size = result["original_size"]
            resized_size = result["resized_size"]

            total_original += original_size
            total_480p += resized_size
            ratios.append(resized_size / original_size)

            if result["was_downscaled"]:
                downscaled_count += 1

        except Exception as e:
            failed += 1
            print(f"[WARN] Failed: {name} | {e}")

        if checked % DOWNLOAD_PROGRESS_EVERY == 0:
            saved = total_original - total_480p
            saved_pct = (saved / total_original * 100) if total_original else 0

            print(
                f"[PROGRESS] checked={checked:,}/{len(sample):,}, "
                f"valid={valid_images:,}, failed={failed:,}, "
                f"original={human_size(total_original)}, "
                f"480p={human_size(total_480p)}, "
                f"saved={human_size(saved)} ({saved_pct:.1f}%)"
            )

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print(f"Total blobs seen: {total_seen:,}")
    print(f"JPEG candidates: {jpeg_candidates:,}")
    print(f"Sampled files: {len(sample):,}")
    print(f"Valid JPEG images: {valid_images:,}")
    print(f"Failed / not images: {failed:,}")
    print(f"Actually reduced files: {downscaled_count:,}")

    print("\nSample size:")
    print(f"  Original: {human_size(total_original)}")
    print(f"  480p JPEG: {human_size(total_480p)}")

    saved = total_original - total_480p
    saved_pct = (saved / total_original * 100) if total_original else 0
    ratio = total_480p / total_original if total_original else 0

    print("\nSavings on sample:")
    print(f"  Saved: {human_size(saved)}")
    print(f"  Saved percent: {saved_pct:.1f}%")
    print(f"  New/original ratio: {ratio:.3f}")
    print(f"  Avg per-file ratio: {mean(ratios):.3f}" if ratios else "  Avg per-file ratio: n/a")

    container_tib = 4.38
    projected_480p_tib = container_tib * ratio
    projected_saved_tib = container_tib - projected_480p_tib

    print("\nProjected for your 4.38 TiB container:")
    print(f"  Current: {container_tib:.2f} TiB")
    print(f"  480p estimate: {projected_480p_tib:.2f} TiB")
    print(f"  Estimated saved: {projected_saved_tib:.2f} TiB")
    print(f"  Estimated saved percent: {saved_pct:.1f}%")

    print("\nPower BI note:")
    print("  Output stays JPEG, so compatibility risk is much lower than HEIC/WebP.")

if __name__ == "__main__":
    main()