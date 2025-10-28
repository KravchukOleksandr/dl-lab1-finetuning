cmd = f"""
set -e

echo "=== STEP 1: Export TensorRT engine from provided model ==="
mkdir -p ./outputs

echo "Running YOLO export..."
yolo export model=${{inputs.model}} format=engine device=0 imgsz=640 dynamic=False half=True nms=True project=./outputs name=export_trt || true

echo "=== STEP 2: Locate produced .engine file ==="
ENGINE_FOUND=$(find ./outputs -maxdepth 4 -type f -iname '*.engine' | head -n 1 || true)
echo "ENGINE_FOUND=$ENGINE_FOUND"

if [ -n "$ENGINE_FOUND" ]; then
    cp "$ENGINE_FOUND" ./outputs/model.engine || true
else
    echo "WARNING: no .engine file found"
fi

echo "=== STEP 3: List ./outputs ==="
ls -R ./outputs || true

echo "Done."
"""