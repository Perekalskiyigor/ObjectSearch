# from ultralytics import YOLO

# model = YOLO("runs/detect/train-4/weights/best.pt")

# # Проверяем изображение
# results = model(
#     "dataset/images/train/2d187b70-71a2-4a70-b761-dd880b8f65b1.jpg",
#     save=True,
#     conf=0.25
# )

# print(results[0].boxes)

from ultralytics import YOLO
import os

def fix_label_files(directory):
    """Заменяет class_id 2 на 1 (tip)"""
    fixed_count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            modified = False
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts and len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id == 2:
                        parts[0] = '1'  # меняем 2 на 1 (tip)
                        new_lines.append(' '.join(parts) + '\n')
                        modified = True
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            
            if modified:
                with open(filepath, 'w') as f:
                    f.writelines(new_lines)
                fixed_count += 1
    
    return fixed_count

# 1. ИСПРАВЛЯЕМ РАЗМЕТКУ
print("🔧 Исправляю метки...")
train_labels = "E:/DEV/ObjectSearch/dataset/labels/train"
val_labels = "E:/DEV/ObjectSearch/dataset/labels/val"

fixed_train = fix_label_files(train_labels) if os.path.exists(train_labels) else 0
fixed_val = fix_label_files(val_labels) if os.path.exists(val_labels) else 0
print(f"✅ Исправлено файлов: train={fixed_train}, val={fixed_val}")

# 2. ПРОВЕРЯЕМ, ЧТО ФАЙЛ data.yaml СУЩЕСТВУЕТ И ПРАВИЛЬНО НАСТРОЕН
data_yaml_path = "data.yaml"
if not os.path.exists(data_yaml_path):
    print(f"❌ ОШИБКА: Файл {data_yaml_path} не найден!")
    print("Создайте файл data.yaml со следующим содержимым:")
    print("""
    path: E:/DEV/ObjectSearch/dataset  # путь к корню датасета
    train: images/train                # путь к тренировочным изображениям
    val: images/val                    # путь к валидационным изображениям
    
    nc: 2                              # количество классов (bolt=0, tip=1)
    names: ['bolt', 'tip']             # имена классов
    """)
    exit()

# 3. ЗАГРУЖАЕМ МОДЕЛЬ И ОБУЧАЕМ
model = YOLO("yolov8n.pt")  # берем предобученную модель

# ОБУЧЕНИЕ НА ВАШИХ ДАННЫХ
results = model.train(
    data="data.yaml",      # путь к файлу с описанием датасета
    epochs=30,             # количество эпох
    imgsz=640,             # размер изображений
    batch=4,               # размер батча
    device='cpu',          # используйте 'cuda' если есть GPU
    workers=0,             # для Windows лучше 0
    project='runs/detect', # папка для сохранения результатов
    name='train-5',        # имя эксперимента
    exist_ok=True,         # перезаписывать существующие результаты
    verbose=True           # показывать подробности обучения
)

print("✅ Обучение завершено!")

# 4. ПРОВЕРЯЕМ РАБОТУ МОДЕЛИ
print("\n🔍 Проверяю модель на тестовом изображении...")
test_image = "E:/DEV/ObjectSearch/dataset/images/train/2d187b70-71a2-4a70-b761-dd880b8f65b1.jpg"

if os.path.exists(test_image):
    # Загружаем лучшую модель
    best_model = YOLO("runs/detect/train-5/weights/best.pt")
    
    results_test = best_model(
        test_image,
        save=True,
        conf=0.25,
        project='runs/detect',
        name='test-results'
    )
    
    # Проверяем, нашла ли модель объекты
    if len(results_test[0].boxes) > 0:
        print(f"✅ Найдено объектов: {len(results_test[0].boxes)}")
        print(f"📊 Классы: {results_test[0].boxes.cls}")
        print(f"📈 Уверенность: {results_test[0].boxes.conf}")
    else:
        print("❌ Модель не нашла объектов на тестовом изображении!")
        print("Возможные причины:")
        print("  - Неправильные пути в data.yaml")
        print("  - В разметке нет объектов (пустые .txt файлы)")
        print("  - Неправильный формат координат в .txt файлах")
else:
    print(f"❌ Тестовое изображение не найдено: {test_image}")