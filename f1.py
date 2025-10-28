cmd = f"""
set -e

echo "=== STEP 1: Prepare model locally ==="
# Copy model weights from the readonly mount into working directory
cp ${{inputs.model}} ./model.pt
ls -l ./model.pt || true

mkdir -p ./outputs

echo "=== STEP 2: Export TensorRT engine ==="
# Export YOLOv8 model to TensorRT from the local copy (./model.pt)
yolo export model=./model.pt format=engine device=0 imgsz=640 dynamic=False half=True project=./outputs name=export_trt --workers 1 || true

echo "=== STEP 3: Locate produced .engine file ==="
ENGINE_FOUND=$(find ./outputs -maxdepth 4 -type f -iname '*.engine' | head -n 1 || true)
echo "ENGINE_FOUND=$ENGINE_FOUND"

if [ -n "$ENGINE_FOUND" ]; then
    cp "$ENGINE_FOUND" ./outputs/model.engine || true
else
    echo "WARNING: no .engine file found"
fi

echo "=== STEP 4: List ./outputs ==="
ls -R ./outputs || true

echo "Done."
"""