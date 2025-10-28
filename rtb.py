CMD_SCRIPT = f"""
set -e

echo "=== DEBUG: list /mnt ==="
ls -R /mnt || true
echo "=== DEBUG: list current dir ==="
pwd
ls -R . || true

# Try to locate mounted model weights (.pt)
echo "=== STEP 1: locate .pt file from inputs ==="
PT_SRC=$(find /mnt -maxdepth 5 -type f -iname "*.pt" | head -n 1 || true)

if [ -z "$PT_SRC" ]; then
    echo "FATAL: no .pt file found under /mnt"
    echo "Listing /mnt for debugging:"
    ls -R /mnt || true
    # we still exit 1 here because без весов дальше нет смысла
    exit 1
fi

echo "Found model weights at: $PT_SRC"

OUTPUT_DIR="./outputs"
mkdir -p "$OUTPUT_DIR"

PT_PATH="$OUTPUT_DIR/best.pt"
cp "$PT_SRC" "$PT_PATH"
echo "Copied weights to: $PT_PATH"

echo "=== STEP 2: export TensorRT engine ==="
yolo export model="$PT_PATH" format=engine device=0 imgsz={IMGSZ} dynamic=False half=True nms=True project="$OUTPUT_DIR" name="export_trt" || {{
    echo "WARNING: yolo export failed. Will still dump outputs dir for debugging."
}}

echo "=== DEBUG: after export, list OUTPUT_DIR ==="
ls -R "$OUTPUT_DIR" || true
ls -R . || true

echo "=== STEP 3: locate .engine ==="
ENGINE_FOUND=$(find "$OUTPUT_DIR" . -maxdepth 6 -type f -iname "*.engine" | head -n 1 || true)

if [ -z "$ENGINE_FOUND" ]; then
    echo "WARNING: no .engine file found after export"
else
    echo "Found engine at: $ENGINE_FOUND"
    cp "$ENGINE_FOUND" "$OUTPUT_DIR/model.engine" || true
    ls -l "$OUTPUT_DIR/model.engine" || true
fi

echo "=== FINAL OUTPUTS DIR ==="
ls -R "$OUTPUT_DIR" || true

echo "Done."
"""