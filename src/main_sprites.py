from __future__ import division
from __future__ import print_function

import os
import sys
import time
from constants import *
from utils_common import *
from datetime import datetime
import tensorflow as tf

from model_sprites import DCGAN
from utils_dcgan import pp

# flags = tf.app.flags
# flags.DEFINE_integer("epoch", 10, "Epoch to train [15]")
# flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
# flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
# flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
# flags.DEFINE_string("checkpoint_dir", "checkpoint_sprites", "Directory name to save the checkpoints [checkpoint]")
# flags.DEFINE_string("summary_dir", "summary_sprites", "Directory name to save the summaries [checkpoint]")
# flags.DEFINE_string("continue_from", None, 'Continues from the given run, None does start training from scratch [None]')
# flags.DEFINE_integer("continue_from_iteration", None,'Continues from the given iteration (of the given run), '
#                                                      'None does restore the most current iteration [None]')
# flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
# FLAGS = flags.FLAGS


def main(argv):
    print('main -->')

    file = [p for p in argv if p == PARAMS_FILE]
    assert len(file) == 1, 'only one params.json allowed'
    params = Params(file[0])
    pp.pprint(params)

    create_dirs(argv, params)

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

        dcgan = DCGAN(sess, batch_size=params.batch_size)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        start_time = time.time()
        dcgan.train(params)
        params.duration = time.time() - start_time

        params.save(os.path.join(params.run_dir, PARAMS_FILE))

    print('main <--')


def create_dirs(argv, params):
    log_dir = params.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    run_dir = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    params.run_dir = run_dir
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    comment = [p[3:] for p in argv if p.startswith('-c=') and len(p[3:]) > 0]
    if comment:
        params.comment = comment[0]

    summary_dir = os.path.join(run_dir, params.summary_dir)
    params.summary_dir = summary_dir
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    checkpoint_dir = os.path.join(run_dir, params.checkpoint_dir)
    params.checkpoint_dir = checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


if __name__ == '__main__':
    tf.app.run(argv=sys.argv[1:])
