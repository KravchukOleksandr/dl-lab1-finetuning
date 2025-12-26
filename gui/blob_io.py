import os
import threading
from dataclasses import dataclass
from azure.storage.blob import BlobServiceClient

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_cache_name(blob_path: str) -> str:
    # avoid collisions between folders
    return blob_path.replace("/", "__")

@dataclass
class AzureBlobIO:
    container_client: any
    filtered_prefix: str

    @staticmethod
    def from_connection_string(conn_str: str, container: str, filtered_prefix: str) -> "AzureBlobIO":
        bsc = BlobServiceClient.from_connection_string(conn_str)
        return AzureBlobIO(bsc.get_container_client(container), filtered_prefix)

    def list_cameras(self) -> list[str]:
        # expects files like meta/cam01_filtered.csv
        cams = []
        for blob in self.container_client.list_blobs(name_starts_with=self.filtered_prefix):
            name = blob.name
            if name.endswith("_filtered.csv"):
                base = os.path.basename(name)
                cams.append(base.replace("_filtered.csv", ""))
        cams = sorted(set(cams))
        return cams

    def download_filtered_csv(self, cam: str, local_path: str) -> None:
        ensure_dir(os.path.dirname(local_path))
        blob_name = f"{self.filtered_prefix}{cam}_filtered.csv"
        bc = self.container_client.get_blob_client(blob_name)
        data = bc.download_blob().readall()
        with open(local_path, "wb") as f:
            f.write(data)

    def download_blob_to(self, blob_path: str, local_path: str) -> None:
        ensure_dir(os.path.dirname(local_path))
        bc = self.container_client.get_blob_client(blob_path)
        data = bc.download_blob().readall()
        with open(local_path, "wb") as f:
            f.write(data)

class BlobCache:
    def __init__(self, blobio: AzureBlobIO, cache_dir_boxes: str, cache_dir_frames: str):
        self.blobio = blobio
        self.cache_boxes = cache_dir_boxes
        self.cache_frames = cache_dir_frames
        ensure_dir(self.cache_boxes)
        ensure_dir(self.cache_frames)

    def box_local_path(self, box_blob: str) -> str:
        return os.path.join(self.cache_boxes, safe_cache_name(box_blob))

    def frame_local_path(self, frame_blob: str) -> str:
        return os.path.join(self.cache_frames, safe_cache_name(frame_blob))

    def get_box(self, box_blob: str) -> str:
        p = self.box_local_path(box_blob)
        if not os.path.exists(p):
            self.blobio.download_blob_to(box_blob, p)
        return p

    def get_frame(self, frame_blob: str) -> str:
        p = self.frame_local_path(frame_blob)
        if not os.path.exists(p):
            self.blobio.download_blob_to(frame_blob, p)
        return p

class Prefetcher:
    """Very small prefetcher: prefetch next (box,frame) in background."""
    def __init__(self, cache: BlobCache):
        self.cache = cache
        self._lock = threading.Lock()
        self._thread = None

    def prefetch(self, box_blob: str, frame_blob: str):
        def job():
            try:
                self.cache.get_box(box_blob)
                self.cache.get_frame(frame_blob)
            except Exception:
                pass

        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=job, daemon=True)
            self._thread.start()
