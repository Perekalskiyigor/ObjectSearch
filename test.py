from ultralytics import YOLO
import cv2
import os
import numpy as np
from pathlib import Path

def diagnose_model(model_path, image_path):
    """Полная диагностика модели"""
    
    print("="*60)
    print("🔍 ДИАГНОСТИКА МОДЕЛИ")
    print("="*60)
    
    # 1. Проверка модели
    print("\n1️⃣ ПРОВЕРКА МОДЕЛИ:")
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        return
    
    model = YOLO(model_path)
    print(f"✅ Модель загружена")
    print(f"   Классы: {model.names}")
    print(f"   Количество классов: {len(model.names)}")
    
    # 2. Проверка изображения
    print(f"\n2️⃣ ПРОВЕРКА ИЗОБРАЖЕНИЯ:")
    if not os.path.exists(image_path):
        print(f"❌ Изображение не найдено: {image_path}")
        return
    
    img = cv2.imread(image_path)
    print(f"✅ Изображение загружено")
    print(f"   Размер: {img.shape}")
    print(f"   Тип: {img.dtype}")
    
    # 3. Проверка на наличие объектов (разные пороги)
    print(f"\n3️⃣ ПОИСК ОБЪЕКТОВ (разные пороги):")
    for conf in [0.01, 0.05, 0.1, 0.25, 0.5]:
        results = model(image_path, conf=conf, verbose=False)
        boxes = results[0].boxes
        count = len(boxes)
        print(f"   Порог {conf:.2f}: найдено {count} объектов")
        if count > 0:
            for box in boxes:
                cls = int(box.cls)
                conf_val = box.conf[0]
                print(f"      Класс {cls} ({model.names[cls]}): {conf_val:.3f}")
    
    # 4. Если объекты не найдены - пробуем аугментацию
    print(f"\n4️⃣ ПОИСК С АУГМЕНТАЦИЕЙ:")
    results = model(image_path, conf=0.1, augment=True, imgsz=1280)
    boxes = results[0].boxes
    print(f"   Найдено объектов: {len(boxes)}")
    
    # 5. Анализ изображения
    print(f"\n5️⃣ АНАЛИЗ ИЗОБРАЖЕНИЯ:")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = np.sum(edges > 0)
    total_pixels = img.shape[0] * img.shape[1]
    edge_percent = (edge_pixels / total_pixels) * 100
    
    print(f"   Границы на изображении: {edge_pixels} пикселей ({edge_percent:.2f}%)")
    if edge_percent < 1:
        print("   ⚠️ На изображении очень мало контраста!")
    elif edge_percent < 5:
        print("   ⚠️ На изображении мало деталей для обнаружения")
    else:
        print("   ✅ На изображении достаточно деталей")
    
    # 6. Рекомендации
    print(f"\n6️⃣ РЕКОМЕНДАЦИИ:")
    if len(boxes) == 0:
        print("   ❌ Объекты не обнаружены")
        print("   Возможные причины:")
        print("   - На изображении нет объектов, на которых модель обучена")
        print("   - Изображение слишком темное или размытое")
        print("   - Модель не дообучена (не хватило эпох)")
        print("   - Неправильные имена классов в разметке")
        print("\n   Что попробовать:")
        print("   - Увеличить количество эпох обучения (50-100)")
        print("   - Добавить аугментацию данных")
        print("   - Проверить правильность разметки")
    else:
        print("   ✅ Объекты найдены!")
        print("   Попробуйте уменьшить порог уверенности для поиска")
    
    print("\n" + "="*60)

# Запуск диагностики
MODEL_PATH = "C:/Users/m.dubrovin/Documents/GitHub/OpenCV/runs/detect/runs/detect/train-5/weights/best.pt"
IMAGE_PATH = "C:/Users/m.dubrovin/Documents/GitHub/ObjectSearch/detect_image/webcam-photo 17.06.2026 11-29-13.jpg"

diagnose_model(MODEL_PATH, IMAGE_PATH)
