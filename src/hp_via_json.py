from __future__ import division
from __future__ import print_function

import os
import sys
from constants import *
from utils_common import *
from datetime import datetime
import tensorflow as tf

def main(argv):
    print('main -->')
    #pp.pprint(flags.FLAGS.__flags)

    file = [p[len(JSON_FILE_PARAM):] for p in argv if p.startswith(JSON_FILE_PARAM) and len(p[len(JSON_FILE_PARAM):]) > 0]
    assert len(file) <= 1, 'only one params.json allowed'
    if not file:
        file.append(JSON_FILE_DEFAULT)
    file = file[0]
    params = Params(file)
    pp.pprint(params)

    create_dirs(argv, params, file)

    #checkpoint_dir = os.path.join(params.log_dir, params.continue_from, params.checkpoint_folder)
    #checkpoint_dir = os.path.join(os.path.dirname(params.checkpoint_dir), params.continue_from)
    #print(checkpoint_dir)

    params.save(os.path.join(params.run_dir, file))

    print('main <--')


def create_dirs(argv, params, file):
    log_dir = params.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    run_dir = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    params.run_dir = run_dir
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    params.save(os.path.join(params.run_dir, file))
    comment = [p[len(COMMENT_PARAM):] for p in argv if p.startswith(COMMENT_PARAM) and len(p[len(COMMENT_PARAM):]) > 0]
    if comment:
        params.comment = comment[0]

    summary_dir = os.path.join(run_dir, params.summary_folder)
    params.summary_dir = summary_dir
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    checkpoint_dir = os.path.join(run_dir, params.checkpoint_folder)
    params.checkpoint_dir = checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


if __name__ == '__main__':
    tf.app.run(argv=sys.argv)
