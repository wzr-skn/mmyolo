_base_ = './yolov6_s_syncbn_fast_8xb32-400e_coco.py'

# dataset settings
data_root = None

num_last_epochs = 5
max_epochs = 120
num_classes = 1

train_batch_size_per_gpu = 2

model = dict(
    data_preprocessor=dict(
        mean=[127.5, 127.5, 127.5],
        std=[128, 128, 128]),
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(
        initial_assigner=dict(num_classes=num_classes),
        assigner=dict(num_classes=num_classes),
    ))

# The training pipeline of YOLOv6 is basically the same as YOLOv5.
# The difference is that Mosaic and RandomAffine will be closed in the last 15 epochs. # noqa

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    dataset=dict(
        data_root=data_root,
        ann_file='/home/ubuntu/my_datasets/OpenImageV6_CrowdHuman/annotation_crowd_head_train.json',
        data_prefix=dict(img='/media/traindata_ro/users/yl3076/ImageDataSets/OpenImageV6_CrowdHuman/WIDER_train/images/')))

val_dataloader = dict(
    dataset=dict(
        ann_file='/media/traindata_ro/users/yl3076/hollywoodheads/hollywoodhead_val.json',
        data_prefix=dict(img='/media/traindata_ro/users/yl3076/hollywoodheads/JPEGImages/')))

test_dataloader = val_dataloader


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
        switch_pipeline=_base_.train_pipeline_stage2)
]

val_evaluator = dict(
    ann_file='/media/traindata_ro/users/yl3076/hollywoodheads/hollywoodhead_val.json')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
