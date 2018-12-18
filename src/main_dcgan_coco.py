from __future__ import division
from __future__ import print_function

import time
import socket
from utils_common import *
from datetime import datetime
import tensorflow as tf

from model_dcgan_coco import DCGAN


def main(argv):
    file, params = init_main(argv)
    print('main -->')
    get_pp().pprint(params)

    # to run on CPU:
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        dcgan = DCGAN(sess, params=params, batch_size=params.batch_size, epochs=params.epochs, \
                       df_dim=params.num_conv_filters_base, image_shape=[params.image_size, params.image_size, 3])
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        start_time = time.time()
        dcgan.train(params)
        params.duration = round(time.time() - start_time, 2)

        params.save(os.path.join(params.run_dir, file))

    print('main <-- [' + str(params.duration) + 's]')


def init_main(argv):
    file = [p[len(JSON_FILE_PARAM):] for p in argv if
            p.startswith(JSON_FILE_PARAM) and len(p[len(JSON_FILE_PARAM):]) > 0]
    assert len(file) <= 1, 'only one params.json allowed'
    if not file:
        file.append(JSON_FILE_DEFAULT)
    file = file[0]
    params = Params(file)
    plausibilize(params)
    create_dirs(argv, params, file)
    copy_src(params)
    init_logging(params.run_dir, LOG_FILE_NAME)
    return file, params


def create_dirs(argv, params, file):
    log_dir = params.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    run_dir = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    params.run_dir = run_dir
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    comment = [p[len(COMMENT_PARAM):] for p in argv if p.startswith(COMMENT_PARAM) and len(p[len(COMMENT_PARAM):]) > 0]
    if comment:
        params.comment = comment[0]
    pid = os.getpid()
    params.pid = pid
    hostname = socket.gethostname()
    params.hostname = hostname
    start = time.strftime("%b %d %Y %H:%M:%S", time.localtime())
    params.training_start = start
    params.save(os.path.join(params.run_dir, file))

    summary_dir = os.path.join(run_dir, params.summary_folder)
    params.summary_dir = summary_dir
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    checkpoint_dir = os.path.join(run_dir, params.checkpoint_folder)
    params.checkpoint_dir = checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    src_dir = os.path.join(run_dir, 'src')
    if not os.path.exists(src_dir):
        os.makedirs(src_dir)
    params.src_dir = src_dir

def plausibilize(params):
    if params.batch_size % 2 != 0:
        print('ERROR: parameter batch_size must be a multiple of 2')
        sys.exit(-1)

if __name__ == '__main__':
    tf.app.run(argv=sys.argv)
