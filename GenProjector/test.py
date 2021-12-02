"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
import util
# from util.visualizer import Visualizer
# from util import html


opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= 1000:
        break
    if i * opt.batchSize >= 0:
        generated = model(data_i, mode='inference')
        nm = data_i['name']
        for b in range(generated.shape[0]):
            # print('process image... %s' % nm[b])

            images = OrderedDict([
                                  ('input', data_i['input'][b]),
                                  ('fake_image', generated[b]),
                                  ('warped', data_i['warped'][b]),
                                  ('im', data_i['crop'][b])])

            util.save_test_images(images, nm[b])
        print (i)