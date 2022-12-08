_base_ = './rtmdet_l_syncbn_8xb32-300e_coco.py'
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'  # noqa

data_root = None
dataset_type = 'YOLOv5CocoDataset'

deepen_factor = 0.33
widen_factor = 0.5
img_scale = (320, 320)
max_epochs = 120
stage2_num_epochs = 20
interval = 10

train_batch_size_per_gpu = 6
train_num_workers = 10
persistent_workers = True
base_lr = 0.004

model = dict(
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[128, 128, 128],
        bgr_to_rgb=True),
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        # Since the checkpoint includes CUDA:0 data,
        # it must be forced to set map_location.
        # Once checkpoint is fixed, it can be removed.
        # init_cfg=dict(
        #     type='Pretrained',
        #     prefix='backbone.',
        #     checkpoint=checkpoint,
        #     map_location='cpu')
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(num_classes=1, widen_factor=widen_factor)))

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=40,
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=(0.5, 2.0),  # note
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOXMixUp',
        img_scale=img_scale,
        use_cached=True,
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
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
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    dataset=[dict(type=dataset_type,
                  ann_file='/home/ubuntu/my_datasets/OpenImageV6_CrowdHuman/annotation_crowd_head_train.json',
                  data_prefix=dict(img='/media/traindata_ro/users/yl3076/ImageDataSets/OpenImageV6_CrowdHuman/WIDER_train/images/'),
                  classes=["person"],
                  filter_cfg=dict(filter_empty_gt=True, min_size=32),
                  pipeline=train_pipeline),
            dict(type=dataset_type,
                  ann_file='/home/ubuntu/my_datasets/SCUT_HEAD/SCUT_HEAD_Part_B/trainval.json',
                  data_prefix=dict(img='/home/ubuntu/my_datasets/SCUT_HEAD/SCUT_HEAD_Part_B/'),
                  classes=["person"],
                  filter_cfg=dict(filter_empty_gt=True, min_size=32),
                  pipeline=train_pipeline),
             dict(type=dataset_type,
                  ann_file='/home/ubuntu/my_datasets/SCUT_HEAD/SCUT_HEAD_Part_A/trainval.json',
                  data_prefix=dict(img='/home/ubuntu/my_datasets/SCUT_HEAD/SCUT_HEAD_Part_A/'),
                  classes=["person"],
                  filter_cfg=dict(filter_empty_gt=True, min_size=32),
                  pipeline=train_pipeline),
             dict(type=dataset_type,
                  ann_file='/home/ubuntu/my_datasets/brainwash/trainbrainwash_.json',
                  data_prefix=dict(img='/home/ubuntu/my_datasets/brainwash/'),
                  classes=["person"],
                  filter_cfg=dict(filter_empty_gt=True, min_size=32),
                  pipeline=train_pipeline),
             dict(type=dataset_type,
                  ann_file='/home/ubuntu/my_datasets/custom_head_dataset/custom_head_dataset.json',
                  data_prefix=dict(img='/home/ubuntu/my_datasets/custom_head_dataset/'),
                  classes=["person"],
                  filter_cfg=dict(filter_empty_gt=True, min_size=32),
                  pipeline=train_pipeline),
             ])

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='/home/ubuntu/my_datasets/OpenImageV6_CrowdHuman/annotation_crowd_head_train.json',
        data_prefix=dict(img='/media/traindata_ro/users/yl3076/ImageDataSets/OpenImageV6_CrowdHuman/WIDER_train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='/media/traindata_ro/users/yl3076/hollywoodheads/hollywoodhead_val.json',
        data_prefix=dict(img='/media/traindata_ro/users/yl3076/hollywoodheads/JPEGImages/')))

test_dataloader = val_dataloader

# Reduce evaluation time
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='/media/traindata_ro/users/yl3076/hollywoodheads/hollywoodhead_val.json',
    metric='bbox')
test_evaluator = val_evaluator

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

work_dir = '../work_dirs/head_detect/rtmdet_4datasets'
