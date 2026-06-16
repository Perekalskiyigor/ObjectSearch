from ultralytics import YOLO

# Загружаем обученную модель
model = YOLO("runs/detect/train-2/weights/best.pt")

# Проверяем изображение
results = model(
    "dataset/images/train/2d187b70-71a2-4a70-b761-dd880b8f65b1.jpg",
    save=True,
    conf=0.25
)

print(results[0].boxes)