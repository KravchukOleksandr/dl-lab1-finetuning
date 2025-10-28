cmd = f"""
set -e

# Step 0: Prepare data
python3 prepare_data.py --stores {' '.join(val_stores)} --input_dir {input_dir}

# Step 1: Define paths
PT_PATH="{model_placeholder}"
ENGINE_PATH="${{PT_PATH%.pt}}.engine"
ONNX_PATH="${{PT_PATH%.pt}}.onnx"

echo "Using model: $PT_PATH"
echo "Engine path: $ENGINE_PATH"

# Step 2: Check if TensorRT engine already exists
if [ -f "$ENGINE_PATH" ]; then
    echo "Found existing TensorRT engine. Skipping export."
else
    echo "TensorRT engine not found. Exporting..."
    # Check if TensorRT Python module is available
    python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('tensorrt') else 1)
PY
    && yolo export model="$PT_PATH" format=engine device=0 imgsz={imgsz} dynamic=False half=True nms=True \
    || (echo "TensorRT module not found. Falling back to ONNX + trtexec export..." && \
        yolo export model="$PT_PATH" format=onnx imgsz={imgsz} dynamic=False && \
        trtexec --onnx="$ONNX_PATH" --saveEngine="$ENGINE_PATH" --explicitBatch --fp16 )
    echo "Engine successfully created: $ENGINE_PATH"
fi

# Step 3: Validate TensorRT engine
echo "Running validation on TensorRT engine..."
yolo task=detect mode=val model="$ENGINE_PATH" data="{input_dir}/data.yaml" batch={batch_size} device=0 imgsz={imgsz} half=True name="{output_name}_trt"

# Step 4: (Optional) Validate original PyTorch model for comparison
echo "Running validation on original PyTorch model..."
yolo task=detect mode=val model="$PT_PATH" data="{input_dir}/data.yaml" batch={batch_size} device=0 imgsz={imgsz} half=True name="{output_name}_pt"

# Step 5: Upload results
python3 upload_data.py --output_name "{output_name}"

echo "All tasks completed successfully."
"""