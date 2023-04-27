# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union, Optional
from pathlib import Path
from torch import Tensor
import os

import torch
from mmdet.structures import SampleList
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.config import Config, DictAction
from mmengine.runner import load_checkpoint

from mmyolo.registry import MODELS


@MODELS.register_module()
class KnowledgeDistillationSingleStageDetector(SingleStageDetector):
    r"""Implementation of `Distilling the Knowledge in a Neural Network.
       <https://arxiv.org/abs/1503.02531>`_.

       Args:
           teacher_config (str | dict): Config file path
               or the config object of teacher model.
           teacher_ckpt (str, optional): Checkpoint path of teacher model.
               If left as None, the model will not load any weights.
       """

    def __init__(self,
                 backbone: ConfigType = None ,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 teacher_config: ConfigType = None,
                 teacher_ckpt: Optional[str] = None,
                 eval_teacher: bool = True,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.eval_teacher = eval_teacher
        # Build teacher model
        if isinstance(teacher_config, (str, Path)):
            teacher_config = Config.fromfile(teacher_config)
        self.teacher_model = MODELS.build(teacher_config.model)
        current_device = 'cuda'
        self.teacher_model.to(current_device)
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location=current_device)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(batch_inputs)
            out_teacher = self.teacher_model.bbox_head.forward(teacher_x)
        losses = self.bbox_head.loss(x, out_teacher, batch_data_samples)
        return losses

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)