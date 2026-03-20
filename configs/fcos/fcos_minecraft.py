_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
# type='FasterRCNN'

model = dict(
    type= 'FCOS',  # model type,
    bbox_head=dict(
        type= 'FCOSHead',
        num_classes=5,
        in_channels=256,  # Must match neck's out_channels
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],  # Stride for each FPN level
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        )
        
    ),
    backbone= dict(
        type= 'ResNet',
        depth= 50,
        # ...
    ),
    neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024],
    out_channels=128,
    num_outs=4,
    start_level=0,
    end_level=2
)

)

metainfo = dict(
    classes=('cow', 'zombie', 'creeper', 'skeleton', 'pig')
)

dataset_type = 'CocoDataset'
data_root = 'datasets/minecraft/'

img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
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
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix=data_root + 'images/train/',
        pipeline=train_pipeline,
        metainfo=metainfo),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'images/val/',
        pipeline=test_pipeline,
        metainfo=metainfo),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'images/val/',
        pipeline=test_pipeline,
        metainfo=metainfo)
)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox')
fp16 = dict(loss_scale='dynamic')

work_dir = 'artifacts/fcos'
load_from = 'checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco.pth'