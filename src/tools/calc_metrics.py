#!/usr/bin/env python3
''' Calculates metrics including IS and FID.

 Usage:
 python calc_metrics.py --gpu <gpu_id> <images_dir> <realdata_fid_stats_dir> <model_id> <model_iteration> <log_dir>

 author: LZ, 15.01.19
'''
from __future__ import absolute_import, division, print_function
import os
from pathlib import Path
home = str(Path.home())
import sys
sys.path.insert(0, os.path.join(home, 'git', 'TTUR')) # fid.py
sys.path.insert(0, os.path.join(home, 'git', 'improved-gan', 'inception_score')) # inception_score.py

from fid import calculate_fid_given_paths
from inception_score import get_inception_score
import numpy as np

import tensorflow as tf
from scipy.misc import imread
from scipy import linalg
from datetime import datetime
import pathlib
import json


class Params:
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path=None):
        if json_path:
            self.update(json_path)

    def __repr__(self):
        return "Params(\n" + str(self.__dict__) + "\n)"

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, usage="python calc_metrics.py --gpu <gpu_id> <images_dir> <realdata_fid_stats_dir> <model_id> <model_iteration> <log_dir>")
    parser.add_argument("path", type=str, nargs=2,
        help='Path to the generated images or to .npz statistic files')
    parser.add_argument("model", type=str, nargs=1,
        help='The experiment-id of the model')
    parser.add_argument("iteration", type=str, nargs=1,
        help='The iteration of the model')
    parser.add_argument("log_dir", type=str, nargs=1,
                        help='The log dir')
    parser.add_argument("-i", "--inception", type=str, default=None,
        help='Path to Inception model (will be downloaded if not provided)')
    parser.add_argument("--gpu", default="", type=str,
        help='GPU to use (leave blank for CPU only)')
    parser.add_argument("--lowprofile", action="store_true",
        help='Keep only one batch of images in memory at a time. This reduces memory footprint, but may decrease speed slightly.')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # print('os.environ[\'CUDA_VISIBLE_DEVICES\'] = %s' % os.environ['CUDA_VISIBLE_DEVICES'])

    path_to_imgs = args.path[0]
    path_to_imgs = pathlib.Path(path_to_imgs)
    files = list(path_to_imgs.glob('*.jpg')) + list(path_to_imgs.glob('*.png'))
    imgs_list = [imread(str(fn)).astype(np.float32) for fn in files]

    print('calculate inception score...')
    mean, std = get_inception_score(imgs_list)
    print("IS: mean=%s, std=%s" % (str(mean), str(std)))
    print('...done.')

    args.path[0] = np.array(imgs_list)
    print('calculate FID...')
    fid_value = calculate_fid_given_paths(args.path, args.inception, low_profile=args.lowprofile)
    print("FID: ", fid_value)
    print('...done.')

    log_dir = args.log_dir[0]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    params = Params()
    params.model = args.model[0]
    params.model_iteration = args.iteration[0]
    params.model_path = args.path[0]
    params.stats_path = args.path[1]
    params.model_fid = fid_value
    params.exec_time = str(datetime.now())

    time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = "log-" + time + "-" + str(args.iteration[0]) + ".json"
    params.save(os.path.join(log_dir, file_name))


