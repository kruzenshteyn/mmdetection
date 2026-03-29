auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
checkpoint_config = dict(interval=1)
data = dict(
    samples_per_gpu=2,
    test=dict(
        ann_file='datasets/minecraft/annotations/instances_test.json',
        img_prefix='datasets/minecraft/images/test/',
        metainfo=dict(
            classes=(
                'cow',
                'zombie',
                'creeper',
                'skeleton',
                'pig',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                img_scale=(
                    512,
                    512,
                ),
                transforms=[
                    dict(keep_ratio=True, type='Resize'),
                    dict(type='RandomFlip'),
                    dict(
                        mean=[
                            102.9801,
                            115.9465,
                            122.7717,
                        ],
                        std=[
                            1.0,
                            1.0,
                            1.0,
                        ],
                        to_rgb=False,
                        type='Normalize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(keys=[
                        'img',
                    ], type='ImageToTensor'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        type='CocoDataset'),
    train=dict(
        ann_file='datasets/minecraft/annotations/instances_train.json',
        img_prefix='datasets/minecraft/images/train/',
        metainfo=dict(
            classes=(
                'cow',
                'zombie',
                'creeper',
                'skeleton',
                'pig',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(img_scale=(
                512,
                512,
            ), keep_ratio=True, type='Resize'),
            dict(flip_ratio=0.5, type='RandomFlip'),
            dict(
                mean=[
                    102.9801,
                    115.9465,
                    122.7717,
                ],
                std=[
                    1.0,
                    1.0,
                    1.0,
                ],
                to_rgb=False,
                type='Normalize'),
            dict(size_divisor=32, type='Pad'),
            dict(type='DefaultFormatBundle'),
            dict(keys=[
                'img',
                'gt_bboxes',
                'gt_labels',
            ], type='Collect'),
        ],
        type='CocoDataset'),
    val=dict(
        ann_file='datasets/minecraft/annotations/instances_valid.json',
        img_prefix='datasets/minecraft/images/valid/',
        metainfo=dict(
            classes=(
                'cow',
                'zombie',
                'creeper',
                'skeleton',
                'pig',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                img_scale=(
                    512,
                    512,
                ),
                transforms=[
                    dict(keep_ratio=True, type='Resize'),
                    dict(type='RandomFlip'),
                    dict(
                        mean=[
                            102.9801,
                            115.9465,
                            122.7717,
                        ],
                        std=[
                            1.0,
                            1.0,
                            1.0,
                        ],
                        to_rgb=False,
                        type='Normalize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(keys=[
                        'img',
                    ], type='ImageToTensor'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        type='CocoDataset'),
    workers_per_gpu=2)
data_root = 'datasets/minecraft/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
evaluation = dict(interval=1, metric='bbox')
fp16 = dict(loss_scale='dynamic')
img_norm_cfg = dict(
    mean=[
        102.9801,
        115.9465,
        122.7717,
    ], std=[
        1.0,
        1.0,
        1.0,
    ], to_rgb=False)
launcher = 'none'
load_from = 'checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(classes=(
    'cow',
    'zombie',
    'creeper',
    'skeleton',
    'pig',
))
model = dict(
    backbone=dict(depth=50, type='ResNet'),
    bbox_head=dict(
        feat_channels=256,
        in_channels=256,
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=5,
        stacked_convs=4,
        strides=[
            8,
            16,
            32,
            64,
            128,
        ],
        type='FCOSHead'),
    neck=dict(
        end_level=2,
        in_channels=[
            256,
            512,
            1024,
        ],
        num_outs=4,
        out_channels=128,
        start_level=0,
        type='FPN'),
    type='FCOS')
optim_wrapper = dict(
    optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
runner = dict(max_epochs=12, type='EpochBasedRunner')
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        flip=False,
        img_scale=(
            512,
            512,
        ),
        transforms=[
            dict(keep_ratio=True, type='Resize'),
            dict(type='RandomFlip'),
            dict(
                mean=[
                    102.9801,
                    115.9465,
                    122.7717,
                ],
                std=[
                    1.0,
                    1.0,
                    1.0,
                ],
                to_rgb=False,
                type='Normalize'),
            dict(size_divisor=32, type='Pad'),
            dict(keys=[
                'img',
            ], type='ImageToTensor'),
            dict(keys=[
                'img',
            ], type='Collect'),
        ],
        type='MultiScaleFlipAug'),
]
train_cfg = dict(max_epochs=50, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        backend_args=None,
        data_prefix=dict(img='train2017/'),
        data_root='data/coco/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(img_scale=(
        512,
        512,
    ), keep_ratio=True, type='Resize'),
    dict(flip_ratio=0.5, type='RandomFlip'),
    dict(
        mean=[
            102.9801,
            115.9465,
            122.7717,
        ],
        std=[
            1.0,
            1.0,
            1.0,
        ],
        to_rgb=False,
        type='Normalize'),
    dict(size_divisor=32, type='Pad'),
    dict(type='DefaultFormatBundle'),
    dict(keys=[
        'img',
        'gt_bboxes',
        'gt_labels',
    ], type='Collect'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'artifacts/fcos'
