def compute_paint_hist(image, mask):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx*gx + gy*gy)

    paint = (
        (mask > 0)
        & (L > 40) & (L < 220)
        & (grad < 15)
    )

    hist, _, _ = np.histogram2d(
        a[paint], b[paint],
        bins=16,
        range=[[0,256],[0,256]]
    )

    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-6

    return hist