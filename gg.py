cmd = """
set -e

echo "=== STEP 1: Prepare model ==="
cp ${inputs.model} ./model.pt
ls -lh ./model.pt || true
mkdir -p ./outputs

echo "=== STEP 2: Export to TensorRT ==="
yolo export model=./model.pt format=engine device=0 imgsz=640 dynamic=False half=True || true

echo "=== STEP 3: Search for model.engine (deep search) ==="
ENGINE_PATH=$(find /mnt / -type f -name 'model.engine' 2>/dev/null | head -n 1 || true)

if [ -z "$ENGINE_PATH" ]; then
  echo "Waiting for model.engine to appear..."
  for i in {1..10}; do
    sleep 5
    ENGINE_PATH=$(find /mnt / -type f -name 'model.engine' 2>/dev/null | head -n 1 || true)
    if [ -n "$ENGINE_PATH" ]; then break; fi
  done
fi

if [ -n "$ENGINE_PATH" ]; then
  echo "Found model.engine at: $ENGINE_PATH"
  cp "$ENGINE_PATH" ./outputs/model.engine
else
  echo "model.engine not found even after wait."
fi

echo "=== STEP 4: Verify ./outputs ==="
ls -lhR ./outputs || true

echo "=== DONE ==="
"""