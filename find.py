from ultralytics import YOLO
import cv2
import os
import shutil

# Загружаем модель
model = YOLO('C:/Users/m.dubrovin/Documents/GitHub/OpenCV/runs/detect/train-5/weights/best.pt')

# Путь к изображению
image_path = 'C:/Users/m.dubrovin/Documents/GitHub/ObjectSearch/detect_image/webcam-photo 17.06.2026 11-29-13.jpg'

# Папка для сохранения результатов
output_folder = 'C:/Users/m.dubrovin/Documents/GitHub/ObjectSearch/show_res'

# Создаем папку если её нет
os.makedirs(output_folder, exist_ok=True)

# Запускаем инференс с сохранением в указанную папку
results = model(
    image_path, 
    save=True,
    project=output_folder,  # Основная папка
    name='',  # Пустое имя, чтобы сохранялось прямо в show_res
    exist_ok=True  # Перезаписывать существующие файлы
)

# Или более простой способ - сохранить результат вручную
results = model(image_path)
for result in results:
    # Получаем изображение с нарисованными боксами
    annotated_img = result.plot()
    
    # Сохраняем в нужную папку
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, annotated_img)
    print(f"Результат сохранен: {output_path}")

print("Готово!")