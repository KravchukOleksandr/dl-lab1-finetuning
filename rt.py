from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml.entities import Input
import datetime

# заполни эти значения своими
SUBSCRIPTION_ID = "..."         # твоя подписка
RESOURCE_GROUP  = "..."         # твой resource group
WORKSPACE_NAME  = "..."         # твой workspace

COMPUTE_NAME    = "gpu-cluster" # твой compute target (GPU)
ENV_NAME        = "YOLO8@latest"  # окружение, в котором будет job

# путь к твоей обученной модели .pt в workspace data store или локально
PT_MODEL_PATH   = "azureml://datastores/workspaceblobstore/paths/models/best.pt"
# ↑ если твоя модель лежит где-то ещё, поменяй на свой путь. 
# Если ты уже раньше подавал её как Input(..., mode="ro_mount"), используй тот же источник.

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

imgsz = 640
batch_size = 32
data_yaml = "data-datasets/data.yaml"  # путь к твоему data.yaml внутри рантайма

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"trt_validation_{timestamp}"

CMD_SCRIPT = f"""
set -e

echo "Step 0: prep data (if needed)"
# Если у тебя реально нужен prepare_data.py, оставь строку ниже. 
# Если нет, можешь удалить.
# python3 prepare_data.py --stores STORE_LIST --input_dir DATA_DIR

PT_PATH="/mnt/inputs/model/best.pt"
OUTPUT_DIR="./outputs"
ENGINE_PATH="$OUTPUT_DIR/model.engine"
ONNX_PATH="$OUTPUT_DIR/model.onnx"

mkdir -p "$OUTPUT_DIR"

echo "Using model: $PT_PATH"
echo "Engine:      $ENGINE_PATH"

echo "Step 1: export TensorRT engine in this exact environment"
python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('tensorrt') else 1)" \
&& yolo export model="$PT_PATH" format=engine device=0 imgsz={imgsz} dynamic=False half=True nms=True project="$OUTPUT_DIR" name="export_trt" \
|| (echo "TensorRT Python module not found -> fallback to ONNX + trtexec" && \
    yolo export model="$PT_PATH" format=onnx imgsz={imgsz} dynamic=False project="$OUTPUT_DIR" name="export_onnx" && \
    trtexec --onnx "$ONNX_PATH" --saveEngine "$ENGINE_PATH" --explicitBatch --fp16 )

if [ ! -f "$ENGINE_PATH" ]; then
    CANDIDATE_ENGINE=$(find "$OUTPUT_DIR" -maxdepth 2 -type f -iname "*.engine" | head -n 1)
    if [ -n "$CANDIDATE_ENGINE" ]; then
        cp "$CANDIDATE_ENGINE" "$ENGINE_PATH"
    fi
fi

if [ ! -f "$ENGINE_PATH" ]; then
    echo "FATAL: engine was not created"
    exit 1
fi

echo "Step 2: validate TensorRT engine"
yolo task=detect mode=val model="$ENGINE_PATH" data="{data_yaml}" batch={batch_size} device=0 imgsz={imgsz} half=True name="{run_name}_trt"

echo "Step 3: validate original PyTorch model"
yolo task=detect mode=val model="$PT_PATH" data="{data_yaml}" batch={batch_size} device=0 imgsz={imgsz} half=True name="{run_name}_pt"

echo "Step 4: list outputs"
ls -R .
echo 'Done.'
"""

model_input = Input(
    type="uri_file",
    path=PT_MODEL_PATH,
    mode="ro_mount",  # read-only mount
)

job = command(
    code=".",  # текущая папка (должна содержать yolo, data.yaml и т.д.)
    command=CMD_SCRIPT,
    environment=ENV_NAME,         # "YOLO8@latest"
    compute=COMPUTE_NAME,         # "gpu-cluster"
    inputs={
        "model": model_input,
    },
    display_name=run_name,
    experiment_name="tensorrt_validation"
)

returned_job = ml_client.create_or_update(job)
print("Submitted job:", returned_job.name)