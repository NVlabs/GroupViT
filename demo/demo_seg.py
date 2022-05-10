# -------------------------------------------------------------------------
# # Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

import argparse
import os.path as osp
import sys

parentdir = osp.dirname(osp.dirname(__file__))
sys.path.insert(0, parentdir)

import mmcv
import torch
from datasets import build_text_transform
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.image import tensor2imgs
from mmcv.parallel import collate, scatter
from models import build_model
from omegaconf import read_write
from segmentation.datasets import COCOObjectDataset, PascalContextDataset, PascalVOCDataset
from segmentation.evaluation import build_seg_demo_pipeline, build_seg_inference
from utils import get_config, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser('GroupViT demo')
    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        help='path to config file',
    )
    parser.add_argument(
        '--opts',
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument(
        '--vis',
        help='Specify the visualization mode, '
        'could be a list, support "input", "pred", "input_pred", "all_groups", "first_group", "final_group", "input_pred_label"',
        default=None,
        nargs='+')

    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--dataset', default='voc', choices=['voc', 'coco', 'context'], help='dataset classes for visualization')

    parser.add_argument('--input', type=str, help='input image path')
    parser.add_argument('--output_dir', type=str, help='output dir')

    args = parser.parse_args()
    args.local_rank = 0  # compatible with config

    return args


def inference(args, cfg):
    model = build_model(cfg.model)
    model = revert_sync_batchnorm(model)
    model.to(args.device)
    model.eval()

    load_checkpoint(cfg, model, None, None)

    text_transform = build_text_transform(False, cfg.data.text_aug, with_dc=False)
    if args.dataset == 'voc':
        dataset_class = PascalVOCDataset
        seg_cfg = 'segmentation/configs/_base_/datasets/pascal_voc12.py'
    elif args.dataset == 'coco':
        dataset_class = COCOObjectDataset
        seg_cfg = 'segmentation/configs/_base_/datasets/coco_object164k.py'
    elif args.dataset == 'context':
        dataset_class = PascalContextDataset
        seg_cfg = 'segmentation/configs/_base_/datasets/pascal_context.py'
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    with read_write(cfg):
        cfg.evaluate.seg.cfg = seg_cfg
        cfg.evaluate.seg.opts = ['test_cfg.mode=whole']

    seg_model = build_seg_inference(model, dataset_class, text_transform, cfg.evaluate.seg)

    vis_seg(seg_model, args.input, args.output_dir, args.vis)


def vis_seg(seg_model, input_img, output_dir, vis_modes):
    device = next(seg_model.parameters()).device
    test_pipeline = build_seg_demo_pipeline()
    # prepare data
    data = dict(img=input_img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(seg_model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
    with torch.no_grad():
        result = seg_model(return_loss=False, rescale=True, **data)

    img_tensor = data['img'][0]
    img_metas = data['img_metas'][0]
    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    assert len(imgs) == len(img_metas)

    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]

        ori_h, ori_w = img_meta['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

        for vis_mode in vis_modes:
            out_file = osp.join(output_dir, 'vis_imgs', vis_mode, f'{vis_mode}.jpg')
            seg_model.show_result(img_show, img_tensor.to(device), result, out_file, vis_mode)


def main():
    args = parse_args()
    cfg = get_config(args)

    with read_write(cfg):
        cfg.evaluate.eval_only = True

    inference(args, cfg)


if __name__ == '__main__':
    main()
