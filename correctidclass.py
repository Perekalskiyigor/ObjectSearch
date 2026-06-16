import os

def fix_label_files(directory):
    """Заменяет class_id 2 на 1 (tip) или 0 (bolt) в зависимости от контекста"""
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
                        # По умолчанию заменяем на 1 (tip), но вы можете изменить на 0 (bolt)
                        parts[0] = '1'  # или '0'
                        new_lines.append(' '.join(parts) + '\n')
                        modified = True
                        print(f"  Исправлен класс 2 -> 1 в {filename}")
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            
            if modified:
                with open(filepath, 'w') as f:
                    f.writelines(new_lines)
                fixed_count += 1
    
    return fixed_count

# Пути к папкам с разметкой
train_labels = "E:/DEV/ObjectSearch/dataset/labels/train"
val_labels = "E:/DEV/ObjectSearch/dataset/labels/val"

print("🔧 Исправляю train метки...")
fixed_train = fix_label_files(train_labels)

print("🔧 Исправляю val метки...")
fixed_val = fix_label_files(val_labels)

print(f"\n✅ Готово! Исправлено файлов: train={fixed_train}, val={fixed_val}")