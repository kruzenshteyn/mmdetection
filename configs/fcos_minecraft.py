# _base_ = [
#     '../_base_/datasets/coco_detection.py',
#     '../_base_/schedules/schedule_1x.py',
#     '../_base_/default_runtime.py'
# ]
# # type='FasterRCNN'

# model = dict(
#     type= 'FCOS',  # model type,
#     bbox_head=dict(
#         type= 'FCOSHead',
#         num_classes=5,
#         in_channels=256,  # Must match neck's out_channels
#         stacked_convs=4,
#         feat_channels=256,
#         strides=[8, 16, 32, 64, 128],  # Stride for each FPN level
#         loss_cls=dict(
#             type='FocalLoss',
#             use_sigmoid=True,
#             gamma=2.0,
#             alpha=0.25,
#             loss_weight=1.0
#         )
        
#     ),
#     backbone= dict(
#         type= 'ResNet',
#         depth= 50,
#         # ...
#     ),
#     neck = dict(
#     type='FPN',
#     in_channels=[256, 512, 1024],
#     out_channels=128,
#     num_outs=4,
#     start_level=0,
#     end_level=2
# )

# )

# metainfo = dict(
#     classes=('cow', 'zombie', 'creeper', 'skeleton', 'pig')
# )

# dataset_type = 'CocoDataset'
# data_root = 'datasets/minecraft/'

# img_norm_cfg = dict(
#     mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(512, 512),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]

# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_train.json',
#         img_prefix=data_root + 'images/train/',
#         pipeline=train_pipeline,
#         metainfo=metainfo),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_valid.json',
#         img_prefix=data_root + 'images/valid/',
#         pipeline=test_pipeline,
#         metainfo=metainfo),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_test.json',
#         img_prefix=data_root + 'images/test/',
#         pipeline=test_pipeline,
#         metainfo=metainfo)
# )

# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# runner = dict(type='EpochBasedRunner', max_epochs=12)
# checkpoint_config = dict(interval=1)
# evaluation = dict(interval=1, metric='bbox')
# fp16 = dict(loss_scale='dynamic')

# work_dir = 'artifacts/fcos'
# load_from = 'checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth'

# configs/fcos/fcos_minecraft.py
# configs/fcos/fcos_minecraft.py
_base_ = [
    './_base_/models/fcos_r50-caffe_fpn_gn-head_1x_coco.py',
    # '../_base_/models/fcos_r50-caffe_fpn_gn-head_1x_coco.py',
    './_base_/datasets/coco_detection.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]

# === МОДЕЛЬ: 18 классов ===
model = dict(
    type='FCOS',
    bbox_head=dict(
        type='FCOSHead',
        num_classes=18
    )
)

# === Метаданные датасета (18 классов Minecraft) ===
metainfo = dict(
    classes=(
        'minecraft-mobs', 'bee', 'chicken', 'cow', 'creeper', 'enderman', 'fox', 'frog',
        'ghast', 'goat', 'llama', 'pig', 'sheep', 'skeleton', 'spider',
        'turtle', 'wolf', 'zombie'
    ),
    palette=[
        (220, 20, 60), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 165, 0),
        (128, 0, 128), (0, 255, 255), (255, 105, 180), (255, 69, 0), (173, 216, 230),
        (144, 238, 144), (255, 215, 0), (0, 191, 255), (255, 0, 0), (0, 128, 0),
        (128, 128, 128), (255, 140, 0), (0, 0, 128)
    ]
)

data_root = 'datasets/minecraft/'

# === Data loaders (MMDetection 3.x) ===
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/annotations_train.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        metainfo=metainfo,
        pipeline=_base_.train_pipeline
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/annotations_valid.json',
        data_prefix=dict(img='valid/images/'),
        metainfo=metainfo,
        pipeline=_base_.test_pipeline
    )
)

test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/annotations_test.json',
        data_prefix=dict(img='test/images/'),
        metainfo=metainfo,
        pipeline=_base_.test_pipeline
    )
)


# test_dataloader = val_dataloader

# === Циклы обучения ===
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=8, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# === Оптимизатор ===
optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
)

param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    begin=0,
    end=12,
    by_epoch=True,
    milestones=[8, 11],
    gamma=0.1
)

# === Циклы обучения ===
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# === Хуки ===
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50)
)

# === Пути и загрузка pretrained ===
work_dir = 'artifacts/fcos'
load_from = 'checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth'   # скачано через mim