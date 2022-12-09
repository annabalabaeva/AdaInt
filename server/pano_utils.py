import os
import numpy as np
from PIL import Image
import py360convert
from copy import deepcopy

import mmcv
import torch
from mmcv.parallel import collate, scatter

from mmedit.apis import init_model
from mmedit.core import tensor2img
from mmedit.datasets.pipelines import Compose

from ailut import ailut_transform


class AdaintEngine:
    def __init__(self,
                 config_path="../adaint/configs/fivekrgb.py",
                 checkpoint_path="../pretrained/AiLUT-FiveK-sRGB.pth",
                 device_id=0):
        self.model = init_model(
            config_path, checkpoint_path, device=torch.device('cuda', device_id)
        )
        cfg = self.model.cfg
        self.device = next(self.model.parameters()).device  # model device
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
        self.test_pipeline = Compose(cfg.test_pipeline)

    def enhance(self, filename, temp_dir, mode):
        if mode == "single":
            return self.process_single(filename)
        elif mode == "cube":
            return self.process_cube(filename, temp_dir)
        else:
            return None

    def process_single(self, filename):
        data = dict(lq_path=filename)
        data = self.test_pipeline(data)
        data = scatter(collate([data], samples_per_gpu=1), [self.device])[0]

        with torch.no_grad():
            result = self.model(test_mode=True, **data)
        img_tensor = result["output"]
        img = tensor2img(img_tensor)

        save_path = "{}_enhanced{}".format(*os.path.splitext(filename))
        mmcv.imwrite(img, save_path)

        return save_path

    def process_cube(self, filename, temp_dir):
        workdir = os.path.join(temp_dir, "faces")
        os.makedirs(workdir)

        cube_side_paths = self.convert_pano_to_four_images(workdir, filename)

        # prepare data
        data_list = []
        for img_path in cube_side_paths:
            data = dict(lq_path=img_path)
            data = self.test_pipeline(data)
            data_list.append(deepcopy(data))
        data_input = scatter(collate(data_list, samples_per_gpu=4), [self.device])[0]

        # forward the model
        with torch.no_grad():
            result = self.model(test_mode=True, **data_input)
        print(result.keys())

        # final infer
        data = dict(lq_path=filename)
        data = self.test_pipeline(data)
        data = scatter(collate([data], samples_per_gpu=1), [self.device])[0]
        outs = ailut_transform(data["lq"], result['lut'], result['vertices'])
        img = tensor2img(outs.cpu())

        save_path = "{}_enhanced{}".format(*os.path.splitext(filename))
        mmcv.imwrite(img, save_path)

        return save_path

    @staticmethod
    def convert_pano_to_four_images(workdir, filename):
        src_img = np.array(Image.open(filename))
        cube_sides = py360convert.e2c(src_img, face_w=256, cube_format="list")
        list_sides = []
        for i, img in enumerate(cube_sides[:4]):
            fullpath = os.path.join(workdir, "{}.png".format(i))
            Image.fromarray(img).save(fullpath)
            list_sides.append(fullpath)

        return list_sides