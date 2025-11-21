import torch
import timm

def repack_convnext_sequential_to_timm(old_path: str, new_path: str):
    # 1. Грузим старый чекпоинт
    ckpt = torch.load(old_path, map_location="cpu")

    model_id   = ckpt["model_id"]
    state_old  = ckpt["state_dict"]   # тут и backbone, и head
    head_info  = ckpt["head"]
    num_classes = head_info["num_classes"]
    class_names = head_info["classes"]

    # 2. Создаём НОРМАЛЬНУЮ timm-модель с нужным числом классов
    model = timm.create_model(
        model_id,
        pretrained=False,
        num_classes=num_classes
    )

    new_state = model.state_dict()

    # 3. Переносим веса backbone: ключи начинаются с "0."
    for k, v in state_old.items():
        if k.startswith("0."):
            # "0.stem.0.weight" -> "stem.0.weight"
            new_k = k.replace("0.", "", 1)
        elif k == "1.weight":
            new_k = "head.fc.weight"
        elif k == "1.bias":
            new_k = "head.fc.bias"
        else:
            # это не веса модели, а, например, 'head' и т.п. — пропускаем
            continue

        if new_k in new_state:
            new_state[new_k] = v
        else:
            print(f"[WARN] ключ {new_k} нет в новой модели")

    # 4. Сохраняем в новом, аккуратном формате
    torch.save({
        "model_id": model_id,
        "num_classes": num_classes,
        "classes": class_names,
        "model_state": new_state,
    }, new_path)

    print(f"✔ Новый checkpoint сохранён в {new_path}")