footprint = OklabColorFootprint(
    n_L=6,
    n_C=5,
    c_max=0.32,
    sigma=0.10,
    min_fraction=0.10,
    use_bin_centers_for_L=True,
)

print("Number of support colors:", len(footprint.support_colors))

footprint.show_support_palette()

result = footprint.compare_color_ratio(
    image1_bgr=img1,
    mask1=work_mask1,
    image2_bgr=img2,
    mask2=work_mask2,
)

print("ratio 1->2:", result["ratio_12"])
print("ratio 2->1:", result["ratio_21"])
print("final ratio:", result["ratio"])
print("hist1 shape:", result["hist1"].shape)
print("hist2 shape:", result["hist2"].shape)
