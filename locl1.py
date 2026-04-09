import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from romatch import roma_outdoor


# =========================================================
# IO / PREPROCESS
# =========================================================

def load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_with_aspect(img: np.ndarray, max_side: int = 1024):
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    if scale != 1.0:
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img, scale


def pad_to_multiple(img: np.ndarray, multiple: int = 14):
    h, w = img.shape[:2]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple

    pad_bottom = new_h - h
    pad_right = new_w - w

    img_pad = cv2.copyMakeBorder(
        img,
        0, pad_bottom,
        0, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return img_pad, (h, w), (new_h, new_w)


def img_to_tensor(img_rgb: np.ndarray, device: str):
    ten = torch.from_numpy(img_rgb).float() / 255.0
    ten = ten.permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
    return ten


# =========================================================
# COORDINATE CONVERSIONS
# =========================================================

def pixels_to_norm(points_px, H: int, W: int, device: str):
    """
    points_px: [N,2] in pixel coords (x, y)
    returns: [N,2] in [-1, 1]
    """
    pts = torch.as_tensor(points_px, dtype=torch.float32, device=device)
    x = 2.0 * pts[:, 0] / max(W - 1, 1) - 1.0
    y = 2.0 * pts[:, 1] / max(H - 1, 1) - 1.0
    return torch.stack([x, y], dim=-1)


def norm_to_pixels(points_norm: torch.Tensor, H: int, W: int):
    """
    points_norm: [N,2] in [-1,1]
    returns: [N,2] in pixel coords (x, y)
    """
    x = (points_norm[:, 0] + 1.0) * 0.5 * max(W - 1, 1)
    y = (points_norm[:, 1] + 1.0) * 0.5 * max(H - 1, 1)
    return torch.stack([x, y], dim=-1)


# =========================================================
# SAMPLING HELPERS
# =========================================================

def bilinear_sample_map(field_hwc: torch.Tensor, points_px: np.ndarray) -> torch.Tensor:
    """
    field_hwc: [H,W,C]
    points_px: [N,2] in pixel coords
    returns: [N,C]
    """
    H, W, C = field_hwc.shape
    device = field_hwc.device

    grid = pixels_to_norm(points_px, H, W, device).view(1, 1, -1, 2)  # [1,1,N,2]
    field_nchw = field_hwc.permute(2, 0, 1).unsqueeze(0)              # [1,C,H,W]

    sampled = F.grid_sample(
        field_nchw,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # [1,C,1,N]

    sampled = sampled[0, :, 0, :].T  # [N,C]
    return sampled


def bilinear_sample_scalar(field_hw: torch.Tensor, points_px: np.ndarray) -> torch.Tensor:
    """
    field_hw: [H,W]
    points_px: [N,2] in pixel coords
    returns: [N]
    """
    H, W = field_hw.shape
    device = field_hw.device

    grid = pixels_to_norm(points_px, H, W, device).view(1, 1, -1, 2)  # [1,1,N,2]
    field_nchw = field_hw.unsqueeze(0).unsqueeze(0)                    # [1,1,H,W]

    sampled = F.grid_sample(
        field_nchw,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # [1,1,1,N]

    sampled = sampled[0, 0, 0, :]  # [N]
    return sampled


# =========================================================
# PATCH UTILITIES
# =========================================================

def make_patch_grid(x1: int, y1: int, x2: int, y2: int, step: int = 2) -> np.ndarray:
    xs = np.arange(x1, x2, step, dtype=np.float32)
    ys = np.arange(y1, y2, step, dtype=np.float32)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Patch is too small for the chosen grid_step.")
    pts = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
    return pts


def clip_patch_xyxy(x1, y1, x2, y2, W, H):
    x1 = int(np.clip(x1, 0, W - 1))
    y1 = int(np.clip(y1, 0, H - 1))
    x2 = int(np.clip(x2, 1, W))
    y2 = int(np.clip(y2, 1, H))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid patch after clipping.")
    return x1, y1, x2, y2


# =========================================================
# MAIN LOGIC
# =========================================================

def transfer_patch_with_roma(
    img1_rgb: np.ndarray,
    img2_rgb: np.ndarray,
    patch_xyxy: tuple,
    roma_model,
    device: str = "cuda",
    max_side: int = 1024,
    grid_step: int = 2,
    certainty_thresh: float = 0.5,
):
    """
    Returns a dict with everything needed for visualization.
    """

    # 1) Resize
    img1_rs, scale1 = resize_with_aspect(img1_rgb, max_side=max_side)
    img2_rs, scale2 = resize_with_aspect(img2_rgb, max_side=max_side)

    H1_rs, W1_rs = img1_rs.shape[:2]
    H2_rs, W2_rs = img2_rs.shape[:2]

    # 2) Scale patch from original img1 coords to resized img1 coords
    x1, y1, x2, y2 = patch_xyxy
    x1r = int(round(x1 * scale1))
    y1r = int(round(y1 * scale1))
    x2r = int(round(x2 * scale1))
    y2r = int(round(y2 * scale1))

    x1r, y1r, x2r, y2r = clip_patch_xyxy(x1r, y1r, x2r, y2r, W1_rs, H1_rs)

    # 3) Pad both images to multiple of 14
    img1_pad, (H1_before_pad, W1_before_pad), (H1, W1) = pad_to_multiple(img1_rs, 14)
    img2_pad, (H2_before_pad, W2_before_pad), (H2, W2) = pad_to_multiple(img2_rs, 14)

    # 4) Tensors
    ten1 = img_to_tensor(img1_pad, device)
    ten2 = img_to_tensor(img2_pad, device)

    assert ten1.shape[-2] % 14 == 0 and ten1.shape[-1] % 14 == 0
    assert ten2.shape[-2] % 14 == 0 and ten2.shape[-1] % 14 == 0

    # 5) RoMa match
    with torch.no_grad():
        warp, certainty = roma_model.match(ten1, ten2, device=device)

    if warp.dim() == 4:
        warp = warp[0]           # [H1, W1, 4]
    if certainty.dim() == 3:
        certainty = certainty[0] # [H1, W1]

    # Last two channels = predicted coordinates in image 2, normalized [-1, 1]
    flow_A_to_B = warp[..., 2:]  # [H1, W1, 2]

    # 6) Dense points over patch on image 1
    patch_points_A = make_patch_grid(x1r, y1r, x2r, y2r, step=grid_step)

    # 7) Sample mapping and certainty for patch points
    mapped_B_norm = bilinear_sample_map(flow_A_to_B, patch_points_A)   # [N,2]
    mapped_B_px_t = norm_to_pixels(mapped_B_norm, H2, W2)              # [N,2]
    cert_t = bilinear_sample_scalar(certainty, patch_points_A)         # [N]

    mapped_B_px = mapped_B_px_t.detach().cpu().numpy()
    cert = cert_t.detach().cpu().numpy()

    # 8) Valid points:
    #    - certainty enough
    #    - inside actual non-padded area of image 2 resized
    valid = (
        (cert >= certainty_thresh) &
        (mapped_B_px[:, 0] >= 0) & (mapped_B_px[:, 0] < W2_before_pad) &
        (mapped_B_px[:, 1] >= 0) & (mapped_B_px[:, 1] < H2_before_pad)
    )

    patch_points_A_valid = patch_points_A[valid]
    mapped_B_px_valid = mapped_B_px[valid]
    cert_valid = cert[valid]

    if len(mapped_B_px_valid) == 0:
        raise RuntimeError(
            "No valid mapped points survived. "
            "Try lowering certainty_thresh or inspect the image pair."
        )

    # 9) Bounding box of mapped patch in image 2
    xb1 = int(np.floor(mapped_B_px_valid[:, 0].min()))
    yb1 = int(np.floor(mapped_B_px_valid[:, 1].min()))
    xb2 = int(np.ceil(mapped_B_px_valid[:, 0].max()))
    yb2 = int(np.ceil(mapped_B_px_valid[:, 1].max()))

    xb1, yb1, xb2, yb2 = clip_patch_xyxy(xb1, yb1, xb2, yb2, W2_before_pad, H2_before_pad)

    # 10) Original patch content from resized image 1
    patch_rgb = img1_rs[y1r:y2r, x1r:x2r].copy()

    # 11) Sparse transferred patch visualization in image 2 coordinates
    transferred_sparse = np.zeros_like(img2_rs)
    transferred_mask = np.zeros((H2_before_pad, W2_before_pad), dtype=np.uint8)

    xsA = np.clip(np.round(patch_points_A_valid[:, 0]).astype(int), 0, W1_before_pad - 1)
    ysA = np.clip(np.round(patch_points_A_valid[:, 1]).astype(int), 0, H1_before_pad - 1)
    xsB = np.clip(np.round(mapped_B_px_valid[:, 0]).astype(int), 0, W2_before_pad - 1)
    ysB = np.clip(np.round(mapped_B_px_valid[:, 1]).astype(int), 0, H2_before_pad - 1)

    colors = img1_rs[ysA, xsA]
    transferred_sparse[ysB, xsB] = colors
    transferred_mask[ysB, xsB] = 255

    # Make sparse transfer easier to see
    kernel = np.ones((3, 3), np.uint8)
    transferred_mask_vis = cv2.dilate(transferred_mask, kernel, iterations=1)

    transferred_sparse_vis = transferred_sparse.copy()
    yy, xx = np.where(transferred_mask > 0)
    for y, x in zip(yy, xx):
        cv2.circle(
            transferred_sparse_vis,
            (x, y),
            1,
            transferred_sparse[y, x].tolist(),
            -1,
        )

    # 12) Region crop on image 2
    mapped_region_img2 = img2_rs[yb1:yb2, xb1:xb2].copy()
    mapped_region_overlay = mapped_region_img2.copy()

    local_mask = transferred_mask_vis[yb1:yb2, xb1:xb2] > 0
    local_transfer = transferred_sparse_vis[yb1:yb2, xb1:xb2]

    if np.any(local_mask):
        mapped_region_overlay[local_mask] = (
            0.6 * mapped_region_overlay[local_mask] +
            0.4 * local_transfer[local_mask]
        ).astype(np.uint8)

    # 13) Full-image visualizations
    vis1 = img1_rs.copy()
    cv2.rectangle(vis1, (x1r, y1r), (x2r, y2r), (255, 0, 0), 2)

    vis2 = img2_rs.copy()
    cv2.rectangle(vis2, (xb1, yb1), (xb2, yb2), (255, 0, 0), 2)

    # Draw subset of valid mapped points
    draw_step = max(1, len(mapped_B_px_valid) // 1500 + 1)
    for (x, y), c in zip(mapped_B_px_valid[::draw_step], cert_valid[::draw_step]):
        g = int(255 * min(1.0, float(c)))
        r = 255 - g
        cv2.circle(vis2, (int(round(x)), int(round(y))), 1, (0, g, r), -1)

    return {
        "img1_resized": img1_rs,
        "img2_resized": img2_rs,
        "patch_rgb": patch_rgb,
        "vis_patch_on_img1": vis1,
        "vis_matches_on_img2": vis2,
        "mapped_bbox_img2": (xb1, yb1, xb2, yb2),
        "mapped_region_img2": mapped_region_img2,
        "mapped_region_overlay": mapped_region_overlay,
        "transferred_sparse_vis": transferred_sparse_vis,
        "transferred_mask_vis": transferred_mask_vis,
        "patch_points_A_valid": patch_points_A_valid,
        "mapped_B_px_valid": mapped_B_px_valid,
        "certainty_valid": cert_valid,
        "scale1": scale1,
        "scale2": scale2,
    }


# =========================================================
# VISUALIZATION / SAVE
# =========================================================

def save_rgb(path: str, img_rgb: np.ndarray):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def show_results(res, figsize=(16, 10)):
    plt.figure(figsize=figsize)

    plt.subplot(2, 3, 1)
    plt.imshow(res["vis_patch_on_img1"])
    plt.title("Image 1: original patch")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(res["vis_matches_on_img2"])
    plt.title("Image 2: mapped points + bbox")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(res["patch_rgb"])
    plt.title("Patch content from image 1")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(res["mapped_region_img2"])
    plt.title("Region on image 2")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(res["mapped_region_overlay"])
    plt.title("Region on image 2 + transferred patch")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(res["transferred_sparse_vis"])
    plt.title("Transferred patch in image-2 coordinates")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def save_all_results(res, out_dir="roma_patch_vis"):
    os.makedirs(out_dir, exist_ok=True)

    save_rgb(os.path.join(out_dir, "01_img1_patch_box.png"), res["vis_patch_on_img1"])
    save_rgb(os.path.join(out_dir, "02_img2_matches_bbox.png"), res["vis_matches_on_img2"])
    save_rgb(os.path.join(out_dir, "03_patch_rgb.png"), res["patch_rgb"])
    save_rgb(os.path.join(out_dir, "04_img2_region.png"), res["mapped_region_img2"])
    save_rgb(os.path.join(out_dir, "05_img2_region_overlay.png"), res["mapped_region_overlay"])
    save_rgb(os.path.join(out_dir, "06_transferred_sparse.png"), res["transferred_sparse_vis"])

    # mask save
    cv2.imwrite(
        os.path.join(out_dir, "07_transferred_mask.png"),
        res["transferred_mask_vis"]
    )

    print(f"Saved visualizations to: {out_dir}")


# =========================================================
# EXAMPLE ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    # -----------------------------
    # USER SETTINGS
    # -----------------------------
    img1_path = "img1.jpg"
    img2_path = "img2.jpg"

    # Patch on image 1 in ORIGINAL image-1 coordinates
    patch_xyxy = (120, 180, 260, 320)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_side = 1024
    grid_step = 2
    certainty_thresh = 0.5
    out_dir = "roma_patch_vis"

    # -----------------------------
    # LOAD
    # -----------------------------
    img1 = load_rgb(img1_path)
    img2 = load_rgb(img2_path)

    # -----------------------------
    # MODEL
    # -----------------------------
    roma_model = roma_outdoor(device=device)

    # -----------------------------
    # RUN
    # -----------------------------
    res = transfer_patch_with_roma(
        img1_rgb=img1,
        img2_rgb=img2,
        patch_xyxy=patch_xyxy,
        roma_model=roma_model,
        device=device,
        max_side=max_side,
        grid_step=grid_step,
        certainty_thresh=certainty_thresh,
    )

    # -----------------------------
    # SHOW + SAVE
    # -----------------------------
    show_results(res)
    save_all_results(res, out_dir=out_dir)

    print("Mapped bbox on image 2:", res["mapped_bbox_img2"])
    print("Valid matched points:", len(res["mapped_B_px_valid"]))



