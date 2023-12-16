_base_ = './yolov6_s_syncbn_fast_8xb32-400e_coco.py'

# dataset settings
data_root = '/media/traindata/'
dataset_type = 'YOLOv5CocoDataset'

num_last_epochs = 10
max_epochs = 120
num_classes = 6

img_scale = (640, 640)  # width, height
test_img_scale = (1024, 576)  # width, height
deepen_factor = 0.27
widen_factor = 0.375
affine_scale = 0.5
train_batch_size_per_gpu = 8

base_lr = 0.01

metainfo = {
    'classes': ('person', 'bottle', 'chair', 'potted plant', ),
    # 'palette': [
    #     (220, 20, 60),
    # ]
}
test_metainfo = {
    'classes': ('person', 'bottle', 'chair', 'potted plant', 'fake_person', 'camera', ),
    # 'palette': [
    #     (220, 20, 60),
    # ]
}

# only on Val
batch_shapes_cfg = None

model = dict(
    data_preprocessor=dict(
        # use multi+_scale training
        type='PPYOLOEDetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='PPYOLOEBatchRandomResize',
                random_size_range=(320, 1280),
                interval=1,
                size_divisor=32,
                random_interp=True,
                keep_ratio=False)
        ],
        mean=[127.5, 127.5, 127.5],
        std=[128, 128, 128],
        bgr_to_rgb=True),
    backbone=dict(
        _delete_=True,
        type='mmpretrain.RepLKNet',
        arch='31t_shallow',
        out_indices=(1, 2, 3, ),
    ),
    # neck的num_csp_blocks参数也随着backbone层数减少而减少
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor, num_csp_blocks=9,),
    bbox_head=dict(
        # type='YOLOv6Head',
        # head_module=dict(num_classes=num_classes, widen_factor=widen_factor, act_cfg=dict(type='ReLU', inplace=True)),
        # loss_bbox=dict(iou_mode='siou'),

        # rtm head
        type='RTMDetHead',
        head_module=dict(num_classes=num_classes, widen_factor=widen_factor, act_cfg=dict(type='ReLU', inplace=True)),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(_delete_=True, type='mmdet.GIoULoss', loss_weight=2.0),
    ),
    # train_cfg=dict(
    #     initial_epoch=-1,
    #     initial_assigner=dict(num_classes=num_classes),
    #     assigner=dict(
    #         num_classes=num_classes,
    #         # use_ciou=True,
    #         # topk=10,
    #         # alpha=0.5,
    #         # beta=6.0,
    #         # eps=1e-9
    #     ),),
    # rtm assigner
    train_cfg=dict(
        _delete_=True,
        assigner=dict(
            type='BatchDynamicSoftLabelAssigner',
            num_classes=num_classes,
            topk=13,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# The training pipeline of YOLOv6 is basically the same as YOLOv5.
# The difference is that Mosaic and RandomAffine will be closed in the last 15 epochs. # noqa
# pre_transform = [
#     dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
#     dict(type='LoadAnnotations', with_bbox=True)
# ]
#
# train_pipeline = [
#     *pre_transform,
#     dict(
#         type='Mosaic',
#         img_scale=img_scale,
#         pad_val=114.0,
#         pre_transform=pre_transform),
#     # dict(type='Load_4PasetImages',
#     #      class_names=["person", "chair", "fake_person", "camera"],
#     #      base_cls_num=3,
#     #      image_root="../data_paste_add",
#     #      prob_of_copy=[1, 0.5, 0.1, 0.1],
#     #      ICON_FACTOR=0.2,
#     #      to_float32=True,
#     #      ),
#     # dict(type='mmdet.PhotoMetricDistortion', brightness_delta=48),
#     dict(
#         type='YOLOv5RandomAffine',
#         max_rotate_degree=0.0,
#         max_translate_ratio=0.1,
#         scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
#         # img_scale is (width, height)
#         border=(-img_scale[0] // 2, -img_scale[1] // 2),
#         border_val=(114, 114, 114),
#         max_shear_degree=0.0),
#     dict(type='YOLOv5HSVRandomAug'),
#     dict(type='mmdet.RandomFlip', prob=0.5),
#     dict(
#         type='mmdet.PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
#                    'flip_direction'))
# ]
#
# train_pipeline_stage2 = [
#     *pre_transform,
#     dict(type='YOLOv5KeepRatioResize', scale=img_scale),
#     # dict(type='Load_4PasetImages',
#     #      class_names=["person", "chair", "fake_person", "camera"],
#     #      base_cls_num=3,
#     #      image_root="../data_paste_add",
#     #      prob_of_copy=[1, 0.5, 0.1, 0.1],
#     #      ICON_FACTOR=0.2,
#     #      to_float32=True,
#     #      ),
#     # dict(type='mmdet.PhotoMetricDistortion', brightness_delta=48),
#     dict(
#         type='LetterResize',
#         scale=img_scale,
#         allow_scale_up=True,
#         pad_val=dict(img=114)),
#     dict(
#         type='YOLOv5RandomAffine',
#         max_rotate_degree=0.0,
#         max_translate_ratio=0.1,
#         scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
#         max_shear_degree=0.0,
#     ),
#     dict(type='YOLOv5HSVRandomAug'),
#     dict(type='mmdet.RandomFlip', prob=0.5),
#     dict(
#         type='mmdet.PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
#                    'flip_direction'))
# ]

albu_train_transform = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]

pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True)
]

last_transform = [
    dict(type='Load_4PasetImages',
         class_names=["person", "chair", "fake_person", "camera"],
         base_cls_num=3,
         image_root="/home/ubuntu/mmyolo/data_paste_add/",
         prob_of_copy=[0.2, 0.1, 0.1, 0.1],
         ICON_FACTOR=[0.1, 0.3],
         to_float32=True,
         ),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transform,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]
train_pipeline = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.5, 1.5),
        max_aspect_ratio=100,
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *last_transform
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.5, 1.5),
        max_aspect_ratio=100,
        border_val=(114, 114, 114)), *last_transform
]
"""
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=20,  # note
        random_pop=False,  # note
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        # img_scale is (width, height)
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=(0.5, 2.0),
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOv5MixUp',
        use_cached=True,
        random_pop=False,
        max_cached_images=10,
        prob=0.5),
    dict(type='mmdet.PackDetInputs')
]
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.RandomResize',
        scale=img_scale,
        ratio_range=(0.5, 2.0),  # note
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]"""

coco_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    ann_file='coco/coco_half_person_80_train.json',
    data_prefix=dict(img='coco/train2017/images'),
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline)
focus_view_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    ann_file='custom_body_dataset/focus_view.json',
    data_prefix=dict(img='custom_body_dataset/focus_view'),
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline)
full_view_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    ann_file='custom_body_dataset/full_view.json',
    data_prefix=dict(img='custom_body_dataset/full_view'),
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline)
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolov5_collate', use_ms_training=True),
    # collate_fn=dict(type='yolov5_collate'),
    dataset=dict(_delete_=True, type='ConcatDataset', datasets=[coco_dataset, focus_view_dataset, full_view_dataset]))
# train_dataloader = dict(
#     batch_size=train_batch_size_per_gpu,
#     collate_fn=dict(type='yolov5_collate', use_ms_training=True),
#     # collate_fn=dict(type='yolov5_collate'),
#     dataset=dict(
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file='coco/coco_half_person_80_train.json',
#         data_prefix=dict(img='coco/train2017/images'),
#         # filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='YOLOv5KeepRatioResize', scale=test_img_scale),
    dict(
        type='LetterResize',
        scale=test_img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=test_metainfo,
        ann_file='coco/coco_half_person_80_val.json',
        data_prefix=dict(img='coco/val2017/images'),
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg
    ))

test_dataloader = val_dataloader


optim_wrapper = dict(
    optimizer=dict(
        lr=base_lr,
        batch_size_per_gpu=train_batch_size_per_gpu),)

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='cosine',
        lr_factor=0.01*2,
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='auto'))

# rtm learning scheduler
# optimizer
# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
#     paramwise_cfg=dict(
#         norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=1.0e-5,
#         by_epoch=False,
#         begin=0,
#         end=1000),
#     dict(
#         # use cosine lr from 150 to 300 epoch
#         type='CosineAnnealingLR',
#         eta_min=base_lr * 0.05,
#         begin=max_epochs // 2,
#         end=max_epochs,
#         T_max=max_epochs // 2,
#         by_epoch=True,
#         convert_to_iter_based=True),
# ]

# hooks
# default_hooks = dict(
#     param_scheduler=dict(_delete_=True, type='ParamSchedulerHook'),
#     checkpoint=dict(
#         type='CheckpointHook',
#         interval=1,
#         max_keep_ckpts=3,
#         save_best='auto'))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - num_last_epochs,
        switch_pipeline=train_pipeline_stage2)
]

val_evaluator = dict(
    type='mmdet.CocoMetric',
    classwise=True,
    proposal_nums=(100, 1, 10),
    ann_file=data_root + 'coco/coco_half_person_82_camera_fake_personval.json',
    metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

load_from = './work_dirs/body_detect/yolov6_31t_shallow_replknet_multi_scale_rtmloss_assigner_v8_data_aug_coco_halfperson_6cls/epoch_120.pth'
work_dir = './work_dirs/body_detect/yolov6_31t_shallow_replknet_multi_scale_rtmloss_assigner_v8_data_aug_coco_halfperson_6cls_60e_load_from'