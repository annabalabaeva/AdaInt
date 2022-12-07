"""
Super stupid code to test the idea for now.
It reads panorama image of size 256x512, converts in to cube,
takes 4 horizontal sides, save in temp directory, then
process all images with AdaInt model and average results.

1. (IMPORTANT) Need to read images in more smart way
2. Need to average embeddings in more smart way
"""

import os
import argparse
import numpy as np
from PIL import Image

import mmcv
import torch
from mmcv.parallel import collate, scatter

from mmedit.apis import init_model
from mmedit.core import tensor2img
from mmedit.datasets.pipelines import Compose

import py360convert
from shutil import rmtree
from copy import deepcopy

from ailut import ailut_transform


def enhancement_inference(model, src_img_path):
    r"""Inference image with the model.
    Args:
        model (nn.Module): The loaded model.
        img (str): File path of input image.
    Returns:
        Tensor: The predicted enhancement result.
    """
    cube_side_paths = convert_pano_to_four_images(src_img_path)


    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # remove gt from test_pipeline
    keys_to_remove = ['gt', 'gt_path']
    for key in keys_to_remove:
        for pipeline in list(cfg.test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                cfg.test_pipeline.remove(pipeline)
            if 'keys' in pipeline and key in pipeline['keys']:
                pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    cfg.test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)
    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)
    # prepare data
    # data_list = []
    # for img_path in cube_side_paths:
    #     data = dict(lq_path=img_path)
    #     data = test_pipeline(data)
    #     data_list.append(deepcopy(data))
    # data_input = scatter(collate(data_list, samples_per_gpu=4), [device])[0]
    #
    # # forward the model
    # with torch.no_grad():
    #     result = model(test_mode=True, **data_input)
    # print(result.keys())

    # final infer
    data = dict(lq_path=src_img_path)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # outs = ailut_transform(data["lq"], result['lut'], result['vertices'])
    # return outs.cpu()
    with torch.no_grad():
        result = model(test_mode=True, **data)
    return result["output"]




def parse_args():
    parser = argparse.ArgumentParser(description='Enhancement demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_path', help='path to input image file')
    parser.add_argument('save_path', help='path to save enhancement result')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def convert_pano_to_four_images(img_path):
    workdir = os.path.join(os.path.dirname(img_path), "temp")
    if os.path.exists(workdir):
        rmtree(workdir)
    os.makedirs(workdir)

    src_img = np.array(Image.open(img_path))
    cube_sides = py360convert.e2c(src_img, face_w=256, cube_format="list")
    list_sides = []
    for i, img in enumerate(cube_sides[:4]):
        fullpath = os.path.join(workdir, "{}.png".format(i))
        Image.fromarray(img).save(fullpath)
        list_sides.append(fullpath)

    return list_sides


def main():
    args = parse_args()

    if not os.path.isfile(args.img_path):
        raise ValueError('It seems that you did not input a valid '
                         '"image_path".')

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))


    output = enhancement_inference(model, args.img_path)
    output = tensor2img(output)

    mmcv.imwrite(output, args.save_path)


if __name__ == '__main__':
    main()