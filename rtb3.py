cmd = f"""
python3 prepare_data.py --stores {' '.join(val_stores)} --input_dir {input_dir} &&

yolo task=detect mode=val model={model_placeholder} batch={batch_size} device=0 data={input_dir}/data.yaml &&

# Export TensorRT engine from the same model
mkdir -p ./outputs && \
echo 'Exporting TensorRT...' && \
yolo export model={model_placeholder} format=engine device=0 imgsz=640 dynamic=False half=True nms=True project=./outputs name=export_trt && \

# Find .engine file and place it at ./outputs/model.engine
ENGINE_FOUND=$(find ./outputs -maxdepth 4 -type f -iname '*.engine' | head -n 1) && \
echo "ENGINE_FOUND=$ENGINE_FOUND" && \
cp "$ENGINE_FOUND" ./outputs/model.engine && \

echo 'Listing ./outputs:' && \
ls -R ./outputs && \

python3 upload_data.py --output_name {output_name}
"""