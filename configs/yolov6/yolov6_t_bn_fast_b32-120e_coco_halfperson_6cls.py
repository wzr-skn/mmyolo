_base_ = './yolov6_s_syncbn_fast_8xb32-400e_coco.py'

# dataset settings
data_root = '/media/traindata/coco/'

num_last_epochs = 5
max_epochs = 120
num_classes = 4

img_scale = (640, 640)  # width, height
test_img_scale = (1024, 576)  # width, height
deepen_factor = 0.33
widen_factor = 0.375
affine_scale = 0.5
train_batch_size_per_gpu = 60

base_lr = 0.005

metainfo = {
    'classes': ('person', 'bottle', 'chair', 'potted plant', ),
    # 'palette': [
    #     (220, 20, 60),
    # ]
}

# only on Val
batch_shapes_cfg = None

model = dict(
    data_preprocessor=dict(
        # use multi+_scale training
        # type='PPYOLOEDetDataPreprocessor',
        # pad_size_divisor=32,
        # batch_augments=[
        #     dict(
        #         type='PPYOLOEBatchRandomResize',
        #         random_size_range=(320, 800),
        #         interval=1,
        #         size_divisor=32,
        #         random_interp=True,
        #         keep_ratio=False)
        # ],
        mean=[127.5, 127.5, 127.5],
        std=[128, 128, 128],
        bgr_to_rgb=True),
    backbone=dict(arch='P5_no_spp', deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(
        type='YOLOv6Head',
        head_module=dict(num_classes=num_classes, widen_factor=widen_factor, act_cfg=dict(type='ReLU', inplace=True)),
        loss_bbox=dict(iou_mode='siou')),
    train_cfg=dict(
        initial_assigner=dict(num_classes=num_classes),
        assigner=dict(num_classes=num_classes),
    ),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# The training pipeline of YOLOv6 is basically the same as YOLOv5.
# The difference is that Mosaic and RandomAffine will be closed in the last 15 epochs. # noqa
pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True)
]

train_pipeline = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    # dict(type='Load_4PasetImages',
    #      class_names=["person", "chair", "fake_person", "camera"],
    #      base_cls_num=3,
    #      image_root="../data_paste_add",
    #      prob_of_copy=[1, 0.5, 0.1, 0.1],
    #      ICON_FACTOR=0.2,
    #      to_float32=True,
    #      ),
    # dict(type='mmdet.PhotoMetricDistortion', brightness_delta=48),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_translate_ratio=0.1,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114),
        max_shear_degree=0.0),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    # dict(type='Load_4PasetImages',
    #      class_names=["person", "chair", "fake_person", "camera"],
    #      base_cls_num=3,
    #      image_root="../data_paste_add",
    #      prob_of_copy=[1, 0.5, 0.1, 0.1],
    #      ICON_FACTOR=0.2,
    #      to_float32=True,
    #      ),
    # dict(type='mmdet.PhotoMetricDistortion', brightness_delta=48),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_translate_ratio=0.1,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_shear_degree=0.0,
    ),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='coco_half_person_80_train.json',
        data_prefix=dict(img='train2017/images'),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

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
        metainfo=metainfo,
        ann_file='coco_half_person_80_val.json',
        data_prefix=dict(img='val2017/images'),
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
        lr_factor=0.01,
        max_epochs=max_epochs))

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
    ann_file=data_root + 'coco_half_person_82_camera_fake_personval.json',
    metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

load_from = './work_dirs/body_detect/yolov6_s_coco_halfperson/best_coco/person_precision_epoch_39.pth'
work_dir = './work_dirs/body_detect/yolov6_t_coco_halfperson_4cls'