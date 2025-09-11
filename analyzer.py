from pathlib import Path
from PIL import Image
import random

# >>> УКАЖИТЕ ПУТЬ К ПАПКЕ train <<<
TRAIN_DIR = Path("/path/to/train")  # например: Path("/home/user/dataset/train")

RESULT_DIR = TRAIN_DIR.parent / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

k_counts = {}  # счётчики индексов по подпапкам

for subdir in sorted(p for p in TRAIN_DIR.iterdir() if p.is_dir()):
    k_counts[subdir.name] = 0
    for img_path in sorted(subdir.rglob("*")):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in exts:
            continue
        if "copy" in img_path.name.lower():  # 1) игнорируем *Copy*
            continue

        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                w, h = im.size

                # 3) случайные доли для размеров кропа в диапазоне [0.3, 1.0]
                scale_w = random.uniform(0.3, 1.0)
                scale_h = random.uniform(0.3, 1.0)
                cw = max(1, int(w * scale_w))
                ch = max(1, int(h * scale_h))

                # центральный кроп с выбранными размерами
                left = (w - cw) // 2
                top = (h - ch) // 2
                right = left + cw
                bottom = top + ch
                crop = im.crop((left, top, right, bottom))

                # 2) имя subfoldername_k с сохранением исходного расширения, если есть
                k = k_counts[subdir.name]
                ext = img_path.suffix.lower() if img_path.suffix.lower() in exts else ".jpg"
                out_name = f"{subdir.name}_{k}{ext}"
                crop.save(RESULT_DIR / out_name)
                k_counts[subdir.name] += 1

        except Exception as e:
            # можно закомментировать, если не хотите видеть ошибки отдельных файлов
            print(f"skip {img_path}: {e}")

print(f"Готово. Сохранено в: {RESULT_DIR}")