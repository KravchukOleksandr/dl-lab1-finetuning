import argparse, time, statistics, contextlib, torch
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark: YOLOv8n-cls vs ConvNeXtV2-Atto vs MobileNetV4-Conv-S")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu","cuda"])
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--precision", choices=["fp32","fp16","bf16"], default="fp32")
    p.add_argument("--channels-last", action="store_true", dest="channels_last")
    p.add_argument("--pretrained", action="store_true", help="(timm) download pretrained weights")
    p.add_argument("--models", nargs="*", default=["yolov8n-cls","convnextv2_atto","mobilenetv4_conv_small"],
                   help="Subset to run: yolov8n-cls, convnextv2_atto, mobilenetv4_conv_small")
    return p.parse_args()

def make_input(device, batch, size, dtype, channels_last):
    x = torch.randn(batch, 3, size, size, device=device, dtype=dtype)
    if channels_last and device=="cuda":
        x = x.to(memory_format=torch.channels_last)
    return x

@contextlib.contextmanager
def autocast_ctx(device, precision):
    if device == "cuda" and precision in ("fp16","bf16"):
        dtype = torch.float16 if precision=="fp16" else torch.bfloat16
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        yield

def benchmark_one(model, x, device, iters, warmup, precision):
    model.eval()
    model.to(device)
    # cuDNN autotune
    torch.backends.cudnn.benchmark = True
    # прогрев
    with torch.no_grad():
        for _ in range(warmup):
            with autocast_ctx(device, precision):
                y = model(x)
        if device=="cuda":
            torch.cuda.synchronize()
    # замеры
    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            with autocast_ctx(device, precision):
                y = model(x)
            if device=="cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
    lat_ms = [t*1000 for t in times]  # per-batch latency
    p50 = np.percentile(lat_ms, 50)
    p90 = np.percentile(lat_ms, 90)
    mean = float(np.mean(lat_ms))
    # throughput (images/sec)
    imgs = x.shape[0]
    fps = imgs / (mean/1000.0)
    return {"p50_ms": p50, "p90_ms": p90, "mean_ms": mean, "fps": fps}

def build_timm_model(name, pretrained, num_classes=1000):
    import timm
    # num_classes оставляем по умолчанию; для скорости не важно, можно 0,
    # но у некоторых моделей тогда исчезает голова -> немного меняется граф.
    model = timm.create_model(name, pretrained=pretrained)
    return model

def build_ultralytics_cls():
    # Берем классификационную nano-модель. Можно сменить на yolo11n-cls.pt.
    from ultralytics import YOLO
    m = YOLO("yolov8n-cls.pt")  # скачает веса при первом запуске
    # Возьмем внутренний nn.Module для чистого forward (без препроц/постпроц)
    return m.model

def main():
    args = parse_args()
    device = args.device
    if device=="cuda":
        torch.cuda.empty_cache()

    # dtype
    if args.precision == "fp16":
        dtype = torch.float16 if device=="cuda" else torch.float32
    elif args.precision == "bf16":
        dtype = torch.bfloat16 if device=="cuda" else torch.float32
    else:
        dtype = torch.float32

    results = []

    for name in args.models:
        if name == "yolov8n-cls":
            model = build_ultralytics_cls()
            pretty = "YOLOv8n-cls (Ultralytics)"
        elif name == "convnextv2_atto":
            model = build_timm_model("convnextv2_atto", pretrained=args.pretrained)
            pretty = "ConvNeXtV2-Atto (timm)"
        elif name == "mobilenetv4_conv_small":
            model = build_timm_model("mobilenetv4_conv_small", pretrained=args.pretrained)
            pretty = "MobileNetV4-Conv-S (timm)"
        else:
            print(f"Unknown model key: {name}")
            continue

        # готовим вход
        x = make_input(device, args.batch, args.imgsz, dtype, args.channels_last)

        # переносим модель и оптимизируем память
        model = model.to(device)
        if args.channels_last and device=="cuda":
            model = model.to(memory_format=torch.channels_last)

        # бенч
        stats = benchmark_one(model, x, device, args.iters, args.warmup, args.precision)
        results.append((pretty, stats))

        # вывод
        print(f"\n== {pretty} ==")
        print(f"device={device}, dtype={dtype}, batch={args.batch}, size={args.imgsz}")
        print(f"latency mean: {stats['mean_ms']:.2f} ms/batch  "
              f"(p50 {stats['p50_ms']:.2f}, p90 {stats['p90_ms']:.2f})")
        print(f"throughput: {stats['fps']:.1f} img/s")

    # краткая таблица
    if results:
        print("\nSummary:")
        for name, s in results:
            print(f"{name:28s}  mean {s['mean_ms']:.2f} ms  p90 {s['p90_ms']:.2f} ms  FPS {s['fps']:.1f}")

if __name__ == "__main__":
    main()