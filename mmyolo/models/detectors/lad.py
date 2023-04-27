# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union, Optional
from pathlib import Path
from torch import Tensor
import os

import torch
import torch.nn as nn
from mmdet.structures import SampleList
from mmdet.models.detectors.kd_one_stage import KnowledgeDistillationSingleStageDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.config import Config, DictAction
from mmengine.runner import load_checkpoint

from mmyolo.registry import MODELS


@MODELS.register_module()
class LAD(KnowledgeDistillationSingleStageDetector):
    """Implementation of `LAD <https://arxiv.org/pdf/2108.10520.pdf>`_."""

    def __init__(self,
                 backbone: ConfigType = None,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 teacher_backbone: ConfigType = None,
                 teacher_neck: ConfigType = None,
                 teacher_bbox_head: ConfigType = None,
                 teacher_config: ConfigType = None,
                 teacher_ckpt: Optional[str] = None,
                 eval_teacher: bool = True,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(KnowledgeDistillationSingleStageDetector,
              self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.eval_teacher = eval_teacher
        # Build teacher model
        self.teacher_model = nn.Module()
        self.teacher_model.backbone = MODELS.build(teacher_backbone)
        if teacher_neck is not None:
            self.teacher_model.neck = MODELS.build(teacher_neck)
        teacher_bbox_head.update(train_cfg=train_cfg)
        teacher_bbox_head.update(test_cfg=test_cfg)
        self.teacher_model.bbox_head = MODELS.build(teacher_bbox_head)
        current_device = 'cuda'
        self.teacher_model.to(current_device)
        # Build teacher model
        # if isinstance(teacher_config, (str, Path)):
        #     teacher_config = Config.fromfile(teacher_config)
        # self.teacher_model = MODELS.build(teacher_config.model)
        # current_device = 'cuda'
        # self.teacher_model.to(current_device)
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location=current_device)

    @property
    def with_teacher_neck(self):
        """bool: whether the detector has a teacher_neck"""
        return hasattr(self.teacher_model, 'neck') and \
            self.teacher_model.neck is not None

    def extract_teacher_feat(self, img):
        """Directly extract teacher features from the backbone+neck."""
        x = self.teacher_model.backbone(img)
        if self.with_teacher_neck:
            x = self.teacher_model.neck(x)
        return x

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """teacher label assign"""
        # get label assignment from the teacher
        with torch.no_grad():
            x_teacher = self.extract_teacher_feat(batch_inputs)
            outs_teacher = self.teacher_model.bbox_head(x_teacher)
            get_assignment_inputs = outs_teacher + (batch_data_samples['bboxes_labels'],
                                  batch_data_samples['img_metas'])
            label_assignment_results = \
                self.teacher_model.bbox_head.get_label_assignment(
                    *get_assignment_inputs)

        # the student use the label assignment from the teacher to learn
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, label_assignment_results,
                                              batch_data_samples)
        return losses