import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from romatch import roma_outdoor


import cv2
import numpy as np
import torch

def pad_to_multiple_of_14(img_rgb):
    h, w = img_rgb.shape[:2]
    new_h = ((h + 13) // 14) * 14
    new_w = ((w + 13) // 14) * 14

    pad_bottom = new_h - h
    pad_right = new_w - w

    img_pad = cv2.copyMakeBorder(
        img_rgb,
        top=0,
        bottom=pad_bottom,
        left=0,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return img_pad, (h, w), (new_h, new_w)

def load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_with_aspect(img, max_side=1024):
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    if scale != 1.0:
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img, scale


def img_to_tensor(img_rgb, device):
    ten = torch.from_numpy(img_rgb).float() / 255.0
    ten = ten.permute(2, 0, 1).unsqueeze(0).to(device)
    return ten


def pixels_to_norm(points_px, H, W, device):
    """
    points_px: [N, 2] as (x, y) in pixel coordinates
    returns: [N, 2] in [-1, 1]
    """
    pts = torch.as_tensor(points_px, dtype=torch.float32, device=device)
    x = 2.0 * (pts[:, 0] / max(W - 1, 1)) - 1.0
    y = 2.0 * (pts[:, 1] / max(H - 1, 1)) - 1.0
    return torch.stack([x, y], dim=-1)


def norm_to_pixels(points_norm, H, W):
    """
    points_norm: [N, 2] in [-1, 1]
    returns: [N, 2] as (x, y) in pixel coordinates
    """
    x = (points_norm[:, 0] + 1.0) * 0.5 * max(W - 1, 1)
    y = (points_norm[:, 1] + 1.0) * 0.5 * max(H - 1, 1)
    return torch.stack([x, y], dim=-1)


def bilinear_sample_map(field_hwc, points_px):
    """
    field_hwc: torch tensor [H, W, C]
    points_px: np.ndarray [N, 2] in pixel coords (x, y)
    returns: torch tensor [N, C]
    """
    H, W, C = field_hwc.shape
    device = field_hwc.device
    pts_norm = pixels_to_norm(points_px, H, W, device).view(1, 1, -1, 2)
    field_nchw = field_hwc.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
    sampled = F.grid_sample(
        field_nchw,
        pts_norm,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # [1,C,1,N]
    sampled = sampled[0, :, 0, :].T  # [N,C]
    return sampled


def bilinear_sample_scalar(field_hw, points_px):
    """
    field_hw: torch tensor [H, W]
    points_px: np.ndarray [N, 2] in pixel coords
    returns: torch tensor [N]
    """
    H, W = field_hw.shape
    device = field_hw.device
    pts_norm = pixels_to_norm(points_px, H, W, device).view(1, 1, -1, 2)
    field_nchw = field_hw.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    sampled = F.grid_sample(
        field_nchw,
        pts_norm,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # [1,1,1,N]
    sampled = sampled[0, 0, 0, :]  # [N]
    return sampled


def make_patch_grid(x1, y1, x2, y2, step=2):
    """
    Returns dense pixel centers inside patch.
    """
    xs = np.arange(x1, x2, step, dtype=np.float32)
    ys = np.arange(y1, y2, step, dtype=np.float32)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Patch is too small for the chosen step.")
    grid = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
    return grid


def warp_patch_mask_to_img2(img1_rgb, img2_rgb, patch_xyxy, roma_model, device="cuda",
                            max_side=1024, grid_step=2, certainty_thresh=0.5):
    """
    Main function.

    patch_xyxy: (x1, y1, x2, y2) in original img1 coordinates
    Returns dict with visualization-ready arrays.
    """
    # Resize both images with aspect ratio preserved.
    img1_rs, scale1 = resize_with_aspect(img1_rgb, max_side=max_side)
    img2_rs, scale2 = resize_with_aspect(img2_rgb, max_side=max_side)

    H1, W1 = img1_rs.shape[:2]
    H2, W2 = img2_rs.shape[:2]

    # Scale patch to resized image-1 coordinates.
    x1, y1, x2, y2 = patch_xyxy
    x1r = int(round(x1 * scale1))
    y1r = int(round(y1 * scale1))
    x2r = int(round(x2 * scale1))
    y2r = int(round(y2 * scale1))

    x1r = np.clip(x1r, 0, W1 - 1)
    x2r = np.clip(x2r, 1, W1)
    y1r = np.clip(y1r, 0, H1 - 1)
    y2r = np.clip(y2r, 1, H1)

    if x2r <= x1r or y2r <= y1r:
        raise ValueError("Invalid patch after resizing/clipping.")

    ten1 = img_to_tensor(img1_rs, device)
    ten2 = img_to_tensor(img2_rs, device)

    # Dense RoMa match.
    with torch.no_grad():
        warp, certainty = roma_model.match(ten1, ten2, device=device)

    # Remove batch dim if present.
    if warp.dim() == 4:
        warp = warp[0]           # [H, W, 4]
    if certainty.dim() == 3:
        certainty = certainty[0] # [H, W]

    # A->B map is the last two channels of warp.
    flow_A_to_B = warp[..., 2:]  # [H1, W1, 2] in normalized coords of image-2

    # Dense grid over the patch in image-1 coordinates.
    patch_points_A = make_patch_grid(x1r, y1r, x2r, y2r, step=grid_step)

    # Sample mapped locations in image-2.
    mapped_B_norm = bilinear_sample_map(flow_A_to_B, patch_points_A)   # [N,2]
    mapped_B_px_t = norm_to_pixels(mapped_B_norm, H2, W2)              # [N,2]
    cert_t = bilinear_sample_scalar(certainty, patch_points_A)         # [N]

    mapped_B_px = mapped_B_px_t.detach().cpu().numpy()
    cert = cert_t.detach().cpu().numpy()

    # Keep only confident points landing inside image-2.
    valid = (
        (cert >= certainty_thresh) &
        (mapped_B_px[:, 0] >= 0) & (mapped_B_px[:, 0] < W2) &
        (mapped_B_px[:, 1] >= 0) & (mapped_B_px[:, 1] < H2)
    )

    patch_points_A_valid = patch_points_A[valid]
    mapped_B_px_valid = mapped_B_px[valid]
    cert_valid = cert[valid]

    if len(mapped_B_px_valid) == 0:
        raise RuntimeError("No valid mapped points survived. Lower certainty_thresh or inspect inputs.")

    # Bounding box of mapped patch in image-2.
    xb1 = int(np.floor(mapped_B_px_valid[:, 0].min()))
    yb1 = int(np.floor(mapped_B_px_valid[:, 1].min()))
    xb2 = int(np.ceil(mapped_B_px_valid[:, 0].max()))
    yb2 = int(np.ceil(mapped_B_px_valid[:, 1].max()))

    xb1 = np.clip(xb1, 0, W2 - 1)
    yb1 = np.clip(yb1, 0, H2 - 1)
    xb2 = np.clip(xb2, 1, W2)
    yb2 = np.clip(yb2, 1, H2)

    # Warp whole image-1 into coordinates of image-2.
    # flow_A_to_B is defined on image-1 grid and points to image-2 coords.
    # For a visually useful demo we do a simple splat from patch points instead of exact inverse warping.
    patch_h = y2r - y1r
    patch_w = x2r - x1r
    patch_rgb = img1_rs[y1r:y2r, x1r:x2r].copy()

    # Create sparse transferred patch canvas in image-2 coordinates.
    transferred_sparse = np.zeros_like(img2_rs)
    transferred_mask = np.zeros((H2, W2), dtype=np.uint8)

    # Take colors from image1 at valid patch samples and splat into image2.
    xsA = np.clip(np.round(patch_points_A_valid[:, 0]).astype(int), 0, W1 - 1)
    ysA = np.clip(np.round(patch_points_A_valid[:, 1]).astype(int), 0, H1 - 1)
    xsB = np.clip(np.round(mapped_B_px_valid[:, 0]).astype(int), 0, W2 - 1)
    ysB = np.clip(np.round(mapped_B_px_valid[:, 1]).astype(int), 0, H2 - 1)

    colors = img1_rs[ysA, xsA]
    transferred_sparse[ysB, xsB] = colors
    transferred_mask[ysB, xsB] = 255

    # Thicken sparse splat for visualization only.
    kernel = np.ones((3, 3), np.uint8)
    transferred_mask_vis = cv2.dilate(transferred_mask, kernel, iterations=1)
    transferred_sparse_vis = transferred_sparse.copy()

    # Simple color propagation for prettier visualization only.
    idx = np.where(transferred_mask > 0)
    for y, x in zip(idx[0], idx[1]):
        cv2.circle(transferred_sparse_vis, (x, y), 1, transferred_sparse[y, x].tolist(), -1)

    # Crop the mapped region from image-2.
    mapped_region_img2 = img2_rs[yb1:yb2, xb1:xb2].copy()
    mapped_region_overlay = mapped_region_img2.copy()

    # Overlay the transferred sparse patch onto mapped region.
    local_mask = transferred_mask_vis[yb1:yb2, xb1:xb2] > 0
    local_transfer = transferred_sparse_vis[yb1:yb2, xb1:xb2]
    mapped_region_overlay[local_mask] = (
        0.6 * mapped_region_overlay[local_mask] + 0.4 * local_transfer[local_mask]
    ).astype(np.uint8)

    # Visualization images.
    vis1 = img1_rs.copy()
    cv2.rectangle(vis1, (x1r, y1r), (x2r, y2r), (255, 0, 0), 2)

    vis2 = img2_rs.copy()
    cv2.rectangle(vis2, (xb1, yb1), (xb2, yb2), (255, 0, 0), 2)

    # Draw matched points.
    for (x, y), c in zip(mapped_B_px_valid[::max(1, len(mapped_B_px_valid)//1000 + 1)], 
                         cert_valid[::max(1, len(cert_valid)//1000 + 1)]):
        color = (0, int(255 * min(1.0, c)), 255 - int(255 * min(1.0, c)))
        cv2.circle(vis2, (int(round(x)), int(round(y))), 1, color, -1)

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
    }


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
    plt.title("Image 2 region + transferred patch overlay")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(res["transferred_sparse_vis"])
    plt.title("Transferred patch in image-2 coords")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img1 = load_rgb("img1.jpg")
    img2 = load_rgb("img2.jpg")

    # Патч на image 1: x1, y1, x2, y2
    patch_xyxy = (120, 180, 260, 320)

    roma_model = roma_outdoor(device=device)

    res = warp_patch_mask_to_img2(
        img1_rgb=img1,
        img2_rgb=img2,
        patch_xyxy=patch_xyxy,
        roma_model=roma_model,
        device=device,
        max_side=1024,
        grid_step=2,
        certainty_thresh=0.5,
    )

    show_results(res)
