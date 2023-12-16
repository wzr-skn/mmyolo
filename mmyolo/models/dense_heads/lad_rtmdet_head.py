from typing import List, Sequence, Tuple, Union
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine import MessageHub
from mmengine.dist import get_dist_info
from mmengine.model import BaseModule, bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor
from mmdet.structures.bbox import distance2bbox
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig, reduce_mean)
from ..utils import gt_instances_preprocess

from mmyolo.registry import MODELS, TASK_UTILS
from .rtmdet_head import RTMDetHead, RTMDetSepBNHeadModule


@MODELS.register_module()
class LAD_RTMDetHead(RTMDetHead):
    """KD YOLOv6Head head used in `YOLOv6 <https://arxiv.org/pdf/2209.02976>`_.

        Args:
            head_module(ConfigType): Base module used for YOLOv6Head
            prior_generator(dict): Points generator feature maps
                in 2D points-based detectors.
            loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
            loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
            train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
                anchor head. Defaults to None.
            test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
                anchor head. Defaults to None.
            init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
                list[dict], optional): Initialization config dict.
                Defaults to None.
        """

    def __init__(self,
                 head_module: ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.QualityFocalLoss',
                     use_sigmoid=True,
                     beta=2.0,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='mmdet.GIoULoss', loss_weight=2.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):

        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        # rtmdet doesn't need loss_obj
        self.loss_obj = None

    def get_label_assignment(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None)-> dict:

        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        device = cls_scores[0].device

        # If the shape does not equal, generate new one
        if featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = featmap_sizes
            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                featmap_sizes, device=device, with_stride=True)
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)

        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1).contiguous()

        flatten_bboxes = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ], 1)
        flatten_bboxes = flatten_bboxes * self.flatten_priors_train[..., -1,
                                                                    None]
        flatten_bboxes = distance2bbox(self.flatten_priors_train[..., :2],
                                       flatten_bboxes)

        assigned_result = self.assigner(flatten_bboxes.detach(),
                                        flatten_cls_scores.detach(),
                                        self.flatten_priors_train, gt_labels,
                                        gt_bboxes, pad_bbox_flag)
        return assigned_result

    def loss(self,
             x: Tuple[Tensor],
             label_assignment_results,
             batch_data_samples: Union[list,dict]) -> dict:
        """KD Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`], dict): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """

        if isinstance(batch_data_samples, list):
            losses = super().loss(x, batch_data_samples)
        else:
            outs = self(x)
            # Fast version
            loss_inputs = outs + (label_assignment_results,
                                  batch_data_samples['bboxes_labels'],
                                  batch_data_samples['img_metas'])
            losses = self.loss_by_feat(*loss_inputs)

        return losses

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            label_assignment_results,
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """KD Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        device = cls_scores[0].device

        # If the shape does not equal, generate new one
        if featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = featmap_sizes
            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                featmap_sizes, device=device, with_stride=True)
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)

        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1).contiguous()

        flatten_bboxes = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ], 1)
        flatten_bboxes = flatten_bboxes * self.flatten_priors_train[..., -1,
                                                                    None]
        flatten_bboxes = distance2bbox(self.flatten_priors_train[..., :2],
                                       flatten_bboxes)

        labels = label_assignment_results['assigned_labels'].reshape(-1)
        label_weights = label_assignment_results['assigned_labels_weights'].reshape(-1)
        bbox_targets = label_assignment_results['assigned_bboxes'].reshape(-1, 4)
        assign_metrics = label_assignment_results['assign_metrics'].reshape(-1)
        cls_preds = flatten_cls_scores.reshape(-1, self.num_classes)
        bbox_preds = flatten_bboxes.reshape(-1, 4)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        avg_factor = reduce_mean(assign_metrics.sum()).clamp_(min=1).item()

        loss_cls = self.loss_cls(
            cls_preds, (labels, assign_metrics),
            label_weights,
            avg_factor=avg_factor)

        if len(pos_inds) > 0:
            loss_bbox = self.loss_bbox(
                bbox_preds[pos_inds],
                bbox_targets[pos_inds],
                weight=assign_metrics[pos_inds],
                avg_factor=avg_factor)
        else:
            loss_bbox = bbox_preds.sum() * 0

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)
