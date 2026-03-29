readme = """# Minecraft mobs detection: FCOS vs YOLOv8

## Цель
Дообучить детекторы объектов FCOS (MMDetection) и YOLOv8s (Ultralytics) на датасете Minecraft и сравнить качество и скорость.

## Файлы dataset удалены из репозитория
Ссылка на скачивание https://disk.yandex.ru/d/ezpvkg_cdDJnNA

Обязательно!!! переделать dataset в формат yolo. coco_to_yolo

## Структура артефактов
- artifacts/fcos/ — логи и веса FCOS (log.json, epoch_*.pth, latest.pth)
- artifacts/yolo/ — логи и веса YOLO (results.csv, weights/best.pt)
- artifacts/inference/fcos/ — примеры инференса FCOS на изображениях
- artifacts/inference/yolo/ — примеры инференса YOLO на изображениях
- artifacts/videos/fcos_inference.mp4 — инференс FCOS на видео
- artifacts/videos/yolo_inference.mp4 — инференс YOLO на видео
- artifacts/metrics/metrics_comparison.csv — сравнение mAP, mAP_50 и FPS
- artifacts/report.pdf — отчёт с графиками и примерами детекций

## Метрики
См. artifacts/metrics/metrics_comparison.csv.

## Как запустить
1. Подготовить датасет в форматах COCO (для MMDetection) и YOLO (для Ultralytics).

