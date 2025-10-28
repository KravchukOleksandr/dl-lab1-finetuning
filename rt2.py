from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml.entities import Input
import datetime

# -----------------------
# FILL THESE WITH YOUR INFO
# -----------------------
SUBSCRIPTION_ID = "YOUR_SUBSCRIPTION_ID"
RESOURCE_GROUP  = "YOUR_RESOURCE_GROUP"
WORKSPACE_NAME  = "YOUR_WORKSPACE_NAME"

COMPUTE_NAME    = "YOUR_GPU_COMPUTE_NAME"     # e.g. "gpu-cluster"
ENV_NAME        = "YOLO8@latest"              # environment to run in (must have ultralytics+yolo CLI, CUDA, ideally TensorRT)

MODEL_NAME      = "my_custom_model"           # registered Azure ML model name
MODEL_VERSION   = "1"                         # registered Azure ML model version

IMGSZ = 640                                   # fixed inference resolution for the engine
# -----------------------

# Create ML client
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

# We'll mount the registered AzureML model into the job.
# NOTE: type="mlflow_model" vs "uri_folder" vs "custom_model" can vary by how you registered it.
# Most commonly for plain PyTorch weights it's "uri_folder".
model_input = Input(
    path=f"azureml:{MODEL_NAME}:{MODEL_VERSION}",
    type="uri_folder",
    mode="ro_mount",  # mounted read-only
)

# Generate a unique name for this run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"export_trt_{timestamp}"

# This is the command that will run INSIDE the AzureML job container on the GPU node.
# It will:
#  - assume the registered model got mounted under /mnt/inputs/model/
#  - assume weights file is /mnt/inputs/model/best.pt
#  - export TensorRT engine with half=True, nms=True, dynamic=False, imgsz=IMGSZ
#  - copy the resulting engine into ./outputs/model.engine
#  - list outputs
CMD_SCRIPT = f"""
set -e

PT_DIR="/mnt/inputs/model"
PT_PATH="$PT_DIR/best.pt"

OUTPUT_DIR="./outputs"
ENGINE_OUT="$OUTPUT_DIR/model.engine"

mkdir -p "$OUTPUT_DIR"

echo "Model directory: $PT_DIR"
echo "Expecting weights at: $PT_PATH"
echo "Output engine path: $ENGINE_OUT"

if [ ! -f "$PT_PATH" ]; then
    echo "ERROR: expected weights not found at $PT_PATH"
    echo "Contents of mounted model dir:"
    ls -R "$PT_DIR"
    exit 1
fi

echo "Step 1: export TensorRT engine in-place (FP16, static shape, NMS)"
yolo export model="$PT_PATH" format=engine device=0 imgsz={IMGSZ} dynamic=False half=True nms=True project="$OUTPUT_DIR" name="export_trt"

# Ultralytics usually drops *.engine under {project}/{name}/...
CANDIDATE_ENGINE=$(find "$OUTPUT_DIR" -maxdepth 2 -type f -iname "*.engine" | head -n 1)

if [ -z "$CANDIDATE_ENGINE" ]; then
    echo "ERROR: no .engine file produced by yolo export"
    echo "Directory listing for debug:"
    ls -R "$OUTPUT_DIR"
    exit 1
fi

echo "Found engine at: $CANDIDATE_ENGINE"
cp "$CANDIDATE_ENGINE" "$ENGINE_OUT"

echo "Final engine artifact:"
ls -l "$ENGINE_OUT"

echo "All done."
"""

# Create the command job
job = command(
    code=".",               # current repo/folder will be uploaded as the job code snapshot
    command=CMD_SCRIPT,     # the bash script above
    environment=ENV_NAME,   # e.g. "YOLO8@latest"
    compute=COMPUTE_NAME,   # e.g. "gpu-cluster"
    inputs={
        "model": model_input,   # mount registered model
    },
    display_name=run_name,
    experiment_name="export_tensorrt_engine",
)

# Submit job
returned_job = ml_client.create_or_update(job)
print("Submitted job:", returned_job.name)