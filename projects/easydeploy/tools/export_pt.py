import argparse
import os
import warnings
from io import BytesIO

import torch
from mmdet.apis import init_detector
from mmengine.config import ConfigDict
from mmengine.utils.path import mkdir_or_exist

from mmyolo.utils import register_all_modules
from projects.easydeploy.model import DeployModel

warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)
warnings.filterwarnings(action='ignore', category=torch.jit.ScriptWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=ResourceWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--model-only', action='store_true', help='Export model only')
    parser.add_argument(
        '--work-dir', default='./work_dir', help='Path to save export model')
    parser.add_argument(
        '--img-size',
        nargs='+',
        type=int,
        default=[576, 1024],
        help='Image size of height and width')
    parser.add_argument(
        '--output-names',
        default=['bbox_cls_1', 'bbox_cls_2', 'bbox_cls_3', 'bbox_reg_1', 'bbox_reg_2', 'bbox_reg_3'],
        help='Output names')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument(
        '--device', default='cuda:1', help='Device used for inference')
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1
    return args


def build_model_from_cfg(config_path, checkpoint_path, device):
    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()
    return model


def main():
    args = parse_args()
    register_all_modules()

    if args.model_only:
        postprocess_cfg = None
        output_names = args.output_names
    else:
        postprocess_cfg = ConfigDict(
            pre_top_k=args.pre_topk,
            keep_top_k=args.keep_topk,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
            backend=args.backend)
        output_names = ['num_dets', 'boxes', 'scores', 'labels']
    baseModel = build_model_from_cfg(args.config, args.checkpoint, args.device)

    deploy_model = DeployModel(
        baseModel=baseModel, postprocess_cfg=postprocess_cfg)
    deploy_model.eval()

    fake_input = torch.randn(args.batch_size, 3,
                             *args.img_size)

    trace_model = torch.jit.trace(deploy_model.cpu().eval(), (fake_input))
    torch.jit.save(trace_model, args.work_dir)


if __name__ == '__main__':
    main()
