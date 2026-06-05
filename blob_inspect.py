# pip install azure-storage-blob

from azure.storage.blob import BlobServiceClient
from datetime import datetime

# ======================
# CONSTANTS
# ======================
CONNECTION_STRING = "PASTE_CONNECTION_STRING_HERE"
CONTAINER_NAME = "PASTE_CONTAINER_NAME_HERE"

PRINT_EVERY = 10_000
CHECK_INDEX_TAGS = True  # если очень много blob'ов, может работать долго

# ======================
# SCRIPT
# ======================
def format_bytes(size_bytes: int) -> str:
    gib = size_bytes / (1024 ** 3)
    tib = size_bytes / (1024 ** 4)
    return f"{size_bytes:,} bytes | {gib:,.2f} GiB | {tib:,.2f} TiB"


def main():
    print("Start:", datetime.now())
    print("Container:", CONTAINER_NAME)
    print("Checking index tags:", CHECK_INDEX_TAGS)
    print("-" * 60)

    service = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container = service.get_container_client(CONTAINER_NAME)

    total_blobs = 0
    total_bytes = 0
    total_tags = 0
    blobs_with_tags = 0

    blobs_by_tier = {}
    bytes_by_tier = {}

    blobs_by_type = {}
    bytes_by_type = {}

    for blob in container.list_blobs():
        total_blobs += 1
        size = blob.size or 0
        total_bytes += size

        tier = blob.blob_tier or "Unknown"
        blob_type = blob.blob_type or "Unknown"

        blobs_by_tier[tier] = blobs_by_tier.get(tier, 0) + 1
        bytes_by_tier[tier] = bytes_by_tier.get(tier, 0) + size

        blobs_by_type[blob_type] = blobs_by_type.get(blob_type, 0) + 1
        bytes_by_type[blob_type] = bytes_by_type.get(blob_type, 0) + size

        if CHECK_INDEX_TAGS:
            try:
                blob_client = container.get_blob_client(blob.name)
                tags = blob_client.get_blob_tags()
                tag_count = len(tags)

                if tag_count > 0:
                    blobs_with_tags += 1
                    total_tags += tag_count

            except Exception as e:
                print(f"[WARN] Cannot read tags for blob: {blob.name} | {e}")

        if total_blobs % PRINT_EVERY == 0:
            print(
                f"[PROGRESS] blobs={total_blobs:,}, "
                f"size={format_bytes(total_bytes)}, "
                f"with_tags={blobs_with_tags:,}, "
                f"total_tags={total_tags:,}"
            )

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(f"Total blobs: {total_blobs:,}")
    print(f"Total size: {format_bytes(total_bytes)}")

    print(f"Blobs with index tags: {blobs_with_tags:,}")
    print(f"Total index tags: {total_tags:,}")

    print("\nBy access tier:")
    for tier, count in sorted(blobs_by_tier.items()):
        print(f"  {tier}: {count:,} blobs | {format_bytes(bytes_by_tier[tier])}")

    print("\nBy blob type:")
    for blob_type, count in sorted(blobs_by_type.items()):
        print(f"  {blob_type}: {count:,} blobs | {format_bytes(bytes_by_type[blob_type])}")

    print("\nEnd:", datetime.now())


if __name__ == "__main__":
    main()