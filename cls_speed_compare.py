import time, contextlib
import torch, numpy as np

# ---------- utils ----------

def make_input(device, size, dtype, channels_last):
    x = torch.randn(1, 3, size, size, device=device, dtype=dtype)
    if channels_last and device=="cuda":
        x = x.to(memory_format=torch.channels_last)
    return x

@contextlib.contextmanager
def autocast_ctx(device, precision):
    if device == "cuda" and precision in ("fp16","bf16"):
        with torch.autocast(device_type="cuda",
                            dtype=torch.float16 if precision=="fp16" else torch.bfloat16):
            yield
    else:
        yield

def synchronize(device):
    if device=="cuda":
        torch.cuda.synchronize()

def time_forward(model, x, device, iters=100, warmup=20, precision="fp32"):
    model.eval()
    torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        for _ in range(warmup):
            with autocast_ctx(device, precision):
                _ = model(x)
        synchronize(device)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            with autocast_ctx(device, precision):
                _ = model(x)
            synchronize(device)
            times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {"mean_ms": arr.mean(),
            "p50_ms": np.percentile(arr,50),
            "p90_ms": np.percentile(arr,90)}

# ---------- builders ----------

def build_yolo():
    from ultralytics import YOLO
    y = YOLO("yolov8n-cls.pt")  # скачается автоматически
    return y.model

def build_convnext():
    import timm
    return timm.create_model("convnextv2_atto", pretrained=True)

def build_mnv4():
    import timm
    return timm.create_model("mobilenetv4_conv_small", pretrained=True)

# ---------- main ----------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = "fp16" if device=="cuda" else "fp32"
    size = 224
    x = make_input(device, size, 
                   torch.float16 if precision=="fp16" else torch.float32,
                   channels_last=True)

    for name, builder in [("YOLOv8n-cls", build_yolo),
                          ("ConvNeXtV2-Atto", build_convnext),
                          ("MobileNetV4-Conv-S", build_mnv4)]:
        model = builder().to(device)
        if device=="cuda":
            model = model.to(memory_format=torch.channels_last)
        stats = time_forward(model, x, device, iters=100, warmup=20, precision=precision)
        print(f"\n{name} | {size}x{size} | {device} {precision}")
        print(f"mean {stats['mean_ms']:.3f} ms  "
              f"p50 {stats['p50_ms']:.3f}  p90 {stats['p90_ms']:.3f}")

if __name__ == "__main__":
    main()