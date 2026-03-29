import json
import os
from tqdm import tqdm
import argparse

def coco_to_yolo(coco_json_path: str, output_dir: str):
    """
    Конвертирует аннотации из формата COCO в формат YOLO.
    
    Args:
        coco_json_path: путь к файлу _annotations.coco.json
        output_dir: папка, куда сохранить labels и classes.txt
    """
    # Загружаем COCO аннотации
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    
    # Создаем маппинг category_id → yolo_class_id (0, 1, 2...)
    categories = coco['categories']
    cat_id_to_yolo_id = {cat['id']: i for i, cat in enumerate(categories)}
    
    # Создаем директории
    os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    
    # Группируем аннотации по image_id
    ann_by_img = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_by_img:
            ann_by_img[img_id] = []
        ann_by_img[img_id].append(ann)
    
    # Конвертируем каждое изображение
    for img_info in tqdm(coco['images'], desc="Конвертация в YOLO"):
        img_id = img_info['id']
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        # Имя txt файла
        txt_filename = os.path.splitext(file_name)[0] + '.txt'
        txt_path = os.path.join(labels_dir, txt_filename)
        
        if img_id not in ann_by_img:
            # Изображение без объектов — создаём пустой файл
            open(txt_path, 'w').close()
            continue
        
        lines = []
        for ann in ann_by_img[img_id]:
            # Пропускаем crowd-аннотации
            if ann.get('iscrowd', 0) == 1:
                continue
                
            bbox = ann['bbox']  # COCO: [x_min, y_min, width, height]
            cat_id = ann['category_id']
            
            # Преобразование в YOLO формат
            x_center = (bbox[0] + bbox[2] / 2) / width
            y_center = (bbox[1] + bbox[3] / 2) / height
            w_norm = bbox[2] / width
            h_norm = bbox[3] / height
            
            yolo_class = cat_id_to_yolo_id[cat_id]
            
            lines.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # Записываем файл
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    # Создаем classes.txt (нужен для YOLOv8/v5/v7)
    classes_path = os.path.join(output_dir, 'classes.txt')
    with open(classes_path, 'w', encoding='utf-8') as f:
        for cat in categories:
            f.write(f"{cat['name']}\n")
    
    print(f"\n✅ Конвертация завершена!")
    print(f"   Метки сохранены в: {labels_dir}")
    print(f"   Классы сохранены в: {classes_path}")
    print(f"   Обработано изображений: {len(coco['images'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Конвертер COCO → YOLO формат')
    parser.add_argument('coco_json', type=str, help='Путь к COCO JSON файлу (например, instances_train2017.json)')
    parser.add_argument('output_dir', type=str, help='Папка для сохранения результатов (будет создана labels/ и classes.txt)')
    
    args = parser.parse_args()
    
    coco_to_yolo(args.coco_json, args.output_dir)

    #  python coco_to_yolo.py datasets/minecraft/annotations/annotations_valid.json datasets/minecraft/valid