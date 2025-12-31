# Навчання YOLOX та запуск на AM67A з використанням TIDL

## Корисні посилання
- [Документація AM67A](https://software-dl.ti.com/jacinto7/esd/processor-sdk-linux-am67a/11_00_00/exports/edgeai-docs/devices/AM67A/linux/index.html)
- [TIDL tools](https://github.com/TexasInstruments/edgeai-tidl-tools)
- [Edge AI ModelMaker](https://github.com/TexasInstruments/edgeai-tensorlab)

---

## 1. AM67A та TIDL

**AM67A** — edge-AI процесор сімейства **AM6xA** від Texas Instruments, орієнтований на задачі комп’ютерного зору та inference нейромереж.

### Апаратні блоки
- ARM Cortex-A53 (Linux)
- C7x DSP з MMA
- Deep Learning Accelerator (DLA)

Для запуску моделей використовується стек **TIDL (TI Deep Learning)**, який оптимізує нейромережі під DSP/DLA.

### Типовий пайплайн
1. Навчання моделі (x86, GPU)
2. Експорт у формат **ONNX**
3. Компіляція та калібровка через **TIDL tools**
4. Inference на **AM67A**

Результат компіляції:
- `deploy_graph.json`
- `net.bin`
- `param.bin`

---

## 2. Навчання YOLOX (edgeai-modelmaker)

Для навчання використовується репозиторій  
**edgeai-tensorlab / edgeai-modelmaker**.

### Підготовка середовища (x86)
```bash
sudo apt update
sudo apt install -y python3 python3-venv git
git clone https://github.com/TexasInstruments/edgeai-tensorlab.git
cd edgeai-tensorlab/edgeai-modelmaker
python3 -m venv mm_env
source mm_env/bin/activate
pip install -r requirements.txt
```

### Конфігурація та датасет

YOLOX-конфіги:
```
configs/vision/detection/yolox/
```

Формат датасету — **COCO**:
```
datasets/my_dataset/
├── images/
│   ├── train/
│   └── val/
└── annotations/
```

Реєстрація датасету:
```bash
python3 tools/dataset_tools/register_dataset.py   --dataset_name my_dataset   --dataset_root datasets/my_dataset
```

### Навчання та експорт

Навчання:
```bash
python3 scripts/train.py   --config yolox_s_lite.yaml   --dataset my_dataset
```

Експорт у ONNX:
```bash
python3 scripts/export.py   --checkpoint best.pth   --export_format onnx
```

---

## 3. Компіляція та запуск на AM67A

Компіляція моделі:
```bash
python3 tidl_model_import.py   -m model.onnx   -o tidl_artifacts   -d calibration_images
```

Згенеровані TIDL-артефакти копіюються на AM67A та використовуються **Edge AI SDK** для inference з апаратним прискоренням.
