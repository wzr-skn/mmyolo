# Copyright (c) OpenMMLab. All rights reserved.
from .yolo_detector import YOLODetector
from .kd_one_stage import KnowledgeDistillationSingleStageDetector
from .lad import LAD

__all__ = ['YOLODetector', 'KnowledgeDistillationSingleStageDetector']
