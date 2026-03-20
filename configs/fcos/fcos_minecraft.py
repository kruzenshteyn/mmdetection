
_base_ = [
    '../fcos_r50-caffe_fpn_gn-head_1x_coco.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 3. Метаинформация 
metainfo = {
    'classes': ('cow', 'zombie', 'creeper', 'skeleton', 'spider', 'pig'),
    'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 142), (255, 0, 0), (0, 255, 0)]
}

# 2. Базовая модель 
model = dict(
    bbox_head=dict(num_classes=6) # Укажите ваше количество классов
)

# 4. Pipelines с простыми аугментациями и сжатием
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Настройка данных
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(dataset=dict(pipeline=train_pipeline, metainfo=metainfo, 
                            ann_file='datasets/minecraft/annotations/annotations_train.json', 
                            img_prefix='datasets/minecraft/annotations/train/')),
    val=dict(pipeline=test_pipeline, metainfo=metainfo, 
             ann_file='datasets/minecraft/annotations/annotations_valid.json', 
             img_prefix='datasets/minecraft/annotations/valid/'),
        test=dict(pipeline=test_pipeline, metainfo=metainfo, 
                  ann_file='datasets/minecraft/annotations/annotations_test.json', 
                  img_prefix='datasets/minecraft/annotations/test/')
)

# 5. Остальные параметры
checkpoint_config = dict(interval=1)
fp16 = dict(loss_scale='dynamic')

# _base_ = [
#     '../_base_/datasets/coco_detection.py',
#     '../_base_/schedules/schedule_1x.py',
#     '../_base_/default_runtime.py'
# ]

# # ==================== ДАТАСЕТ ====================
# dataset_type = 'CocoDataset'
# data_root = 'datasets/minecraft/'

# metainfo = dict(
#     classes=(
#         'creeper', 'zombie', 'diamond_ore', 'player', 'pig', 'cow', 'sheep'  # ← ЗАМЕНИТЕ НА ВАШИ РЕАЛЬНЫЕ КЛАССЫ!
#         # Посмотрите в файле annotations/instances_train.json -> "categories"
#     )
# )
# num_classes = len(metainfo['classes'])

# # ==================== МОДЕЛЬ (базовая FCOS R50) ====================
# model = dict(
#     type='FCOS',
#     data_preprocessor=dict(
#         type='DetDataPreprocessor',
#         mean=[102.9801, 115.9465, 122.7717],
#         std=[1.0, 1.0, 1.0],
#         bgr_to_rgb=False,
#         pad_size_divisor=32),
#     backbone=dict(
#         type='ResNet',
#         depth=50,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=1,
#         norm_cfg=dict(type='BN', requires_grad=False),
#         norm_eval=True,
#         style='caffe',
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint='open-mmlab://detectron/resnet50_caffe')),
#     neck=dict(
#         type='FPN',
#         in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         start_level=1,
#         add_extra_convs='on_output',
#         num_outs=5,
#         relu_before_extra_convs=True),
#     bbox_head=dict(
#         type='FCOSHead',
#         num_classes=num_classes,          # ← важно!
#         in_channels=256,
#         stacked_convs=4,
#         feat_channels=256,
#         strides=[8, 16, 32, 64, 128],
#         loss_cls=dict(
#             type='FocalLoss',
#             use_sigmoid=True,
#             gamma=2.0,
#             alpha=0.25,
#             loss_weight=1.0),
#         loss_bbox=dict(type='IoULoss', loss_weight=1.0),
#         loss_centerness=dict(
#             type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
#     test_cfg=dict(
#         nms_pre=1000,
#         min_bbox_size=0,
#         score_thr=0.05,
#         nms=dict(type='nms', iou_threshold=0.5),
#         max_per_img=100))

# # ==================== PIPELINES (простые аугментации + 512x512) ====================
# train_pipeline = [
#     dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', scale=(512, 512), keep_ratio=True),   # сжатие по ТЗ
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PackDetInputs')
# ]

# test_pipeline = [
#     dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
#     dict(type='Resize', scale=(512, 512), keep_ratio=True),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='PackDetInputs',
#          meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
# ]

# # ==================== DATA (ресурсы ВМ) ====================
# data = dict(
#     samples_per_gpu=2,      # по ТЗ
#     workers_per_gpu=2,      # по ТЗ
#     train=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='datasets/minecraft/annotations/annotations_train.json',   # ← подставьте точное имя вашего json
#         metainfo=metainfo,
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='datasets/minecraft/annotations/annotations_valid.json',
#         metainfo=metainfo,
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='datasets/minecraft/annotations/annotations_test.json',
#         metainfo=metainfo,
#         pipeline=test_pipeline))

# # ==================== ОПТИМИЗАЦИИ ====================
# fp16 = dict(loss_scale='dynamic')                     # половинная точность

# default_hooks = dict(
#     checkpoint=dict(type='CheckpointHook', interval=1))  # сохранять раз в эпоху