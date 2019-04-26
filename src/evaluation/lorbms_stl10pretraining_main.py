from __future__ import division
from __future__ import print_function

import sys
sys.path.append('..')
import time
import socket
from utils_common import *
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # str(params.gpu)

from datetime import datetime
import tensorflow as tf

from lorbms_stl0pretraining_model import DCGAN

def main(argv):
    params = init_main(argv)
    print('main -->')
    get_pp().pprint(params)

    if params.gpu == -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1" # str(params.gpu)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu)

    print('os.environ[CUDA_VISIBLE_DEVICES] = %s' % os.environ['CUDA_VISIBLE_DEVICES'])

    start_time = time.time()

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        dcgan = DCGAN(sess, params=params, batch_size=params.batch_size, epochs=params.epochs, \
                       df_dim=params.num_conv_filters_base, image_shape=[params.image_size, params.image_size, 3], is_train=params.is_train)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        dcgan.train(params)

    params.duration = round(time.time() - start_time, 2)
    params.save(os.path.join(params.run_dir, JSON_FILE_DEFAULT))

    print('main <-- [' + str(params.duration) + 's]')


def init_main(argv):
    file = [p[len(JSON_FILE_PARAM):] for p in argv if
            p.startswith(JSON_FILE_PARAM) and len(p[len(JSON_FILE_PARAM):]) > 0]
    assert len(file) <= 1, 'only one params.json allowed'
    if not file:
        file.append('params_stl10pretraining.json')
    file = file[0]
    params = Params(file)
    plausibilize(params)
    create_dirs(argv, params)
    copy_src(params)
    init_logging(params.run_dir, LOG_FILE_NAME)
    return params


def create_dirs(argv, params):
    log_dir = params.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    run_dir = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    params.run_dir = run_dir
    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)
    comment = [p[len(COMMENT_PARAM):] for p in argv if p.startswith(COMMENT_PARAM) and len(p[len(COMMENT_PARAM):]) > 0]
    if comment:
        params.comment = comment[0]
    pid = os.getpid()
    params.pid = pid
    hostname = socket.gethostname()
    params.hostname = hostname
    start = time.strftime("%b %d %Y %H:%M:%S", time.localtime())
    params.training_start = start
    params.save(os.path.join(params.run_dir, JSON_FILE_DEFAULT))

    summary_dir = os.path.join(run_dir, params.summary_folder)
    params.summary_dir = summary_dir
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir, exist_ok=True)
    checkpoint_dir = os.path.join(run_dir, params.checkpoint_folder)
    params.checkpoint_dir = checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    src_dir = os.path.join(run_dir, 'src')
    if not os.path.exists(src_dir):
        os.makedirs(src_dir, exist_ok=True)
    params.src_dir = src_dir

    if params.is_train:
        metric_model_dir = os.path.join(run_dir, params.metric_model_folder)
        # extra dir to save model for later FID calculation
        params.metric_model_dir = metric_model_dir
        if not os.path.exists(metric_model_dir):
            os.makedirs(metric_model_dir, exist_ok=True)
    else: # test
        assert params.test_from
        assert params.metric_model_iteration

        metric_fid_dir = os.path.join(params.log_dir, params.test_from, params.metric_fid_folder, params.stats_type, str(params.metric_model_iteration))
        if not os.path.exists(metric_fid_dir):
            os.makedirs(metric_fid_dir, exist_ok=True)
            print('created metric_fid_dir: %s' % metric_fid_dir)
        params.metric_fid_dir = metric_fid_dir
        metric_fid_out_dir = os.path.join(metric_fid_dir, "images")
        if not os.path.exists(metric_fid_out_dir):
            os.makedirs(metric_fid_out_dir, exist_ok=True)
            print('created metric_fid_out_dir: %s' % metric_fid_out_dir)
        params.metric_fid_out_dir = metric_fid_out_dir
        metric_fid_out_dir_all = os.path.join(metric_fid_dir, "images_all")
        if not os.path.exists(metric_fid_out_dir_all):
            os.makedirs(metric_fid_out_dir_all, exist_ok=True)
            print('created metric_fid_out_dir_all: %s' % metric_fid_out_dir_all)
        params.metric_fid_out_dir_all = metric_fid_out_dir_all
        metric_fid_out_mixed_feature = os.path.join(metric_fid_out_dir_all, "mixed_feature")
        if not os.path.exists(metric_fid_out_mixed_feature):
            os.makedirs(metric_fid_out_mixed_feature, exist_ok=True)
            print('created metric_fid_out_mixed_feature: %s' % metric_fid_out_mixed_feature)
        params.metric_fid_out_mixed_feature = metric_fid_out_mixed_feature

        metric_model_dir = os.path.join(params.log_dir, params.test_from, params.metric_model_folder)
        params.metric_model_dir = metric_model_dir
        metric_results_folder = os.path.join(params.log_dir, params.test_from, params.metric_results_folder, params.stats_type)
        if not os.path.exists(metric_results_folder):
            os.makedirs(metric_results_folder, exist_ok=True)
            print('created metric_results_folder: %s' % metric_results_folder)
        params.metric_results_folder = metric_results_folder


def plausibilize(params):
    if params.batch_size % 2 != 0:
        print('ERROR: parameter batch_size must be a multiple of 2')
        sys.exit(-1)
    # params.is_train = ast.literal_eval(params.is_train)
    assert type(params.is_train) == bool

    if params.gpu not in [-1, 0, 1]:
        print('ERROR: parameter gpu not supported, must be one of -1,0,1')
        sys.exit(-1)

    print("is_train: %s" % str(params.is_train))

    if params.is_train:
        params.tfrecords_path = params.train_tfrecords_path
        params.full_imgs_path = params.train_full_imgs_path
    elif params.stats_type == "test" or params.dump_testset_only:
        params.tfrecords_path = params.stats_test_tfrecords_path
        params.full_imgs_path = params.stats_test_full_imgs_path
        params.path_to_stats_npz = params.stats_test_fid_stats_npz
        params.epochs = 1  # for test process each image only once
    elif params.stats_type == "training":
        params.tfrecords_path = params.stats_train_tfrecords_path
        params.full_imgs_path = params.stats_train_full_imgs_path
        params.path_to_stats_npz = params.stats_train_fid_stats_npz
        params.epochs = 1  # for test process each image only once
    else:
        assert 1 == 0, params.stats_type + " not supported."


if __name__ == '__main__':
    tf.app.run(argv=sys.argv)
