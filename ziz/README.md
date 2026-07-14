# HelmetMicroNeXt baseline

## Class mapping

- `0 = helmet` — каска есть
- `1 = no_helmet` — каски нет; это positive class для precision/recall/F1/AUC

## Dataset structure

```text
ziz-crops-202607-2/
├── train/
│   ├── helmet/
│   └── no-helmet/   # also accepted: no_helmet or nohelmet
└── val/
    ├── helmet/
    └── no-helmet/
```

The images are expected to be already expanded by x1.25 and fully inside the
original frame.

## Run

1. Edit `DATASET_ROOT` and, if needed, hyperparameters at the top of `main.py`.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start training:

```bash
python main.py
```

Every epoch prints validation loss, accuracy, precision, recall, F1, ROC-AUC and
confusion-matrix values. The positive class is always class `1 = no_helmet`.

Outputs:

```text
runs/helmet_micronext_v1/
├── best.pt
├── last.pt
└── history.csv
```

The train loader uses a balanced sampler. `BCEWithLogitsLoss` therefore uses no
additional `pos_weight`.
