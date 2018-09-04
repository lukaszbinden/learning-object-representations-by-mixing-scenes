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

    file = [p for p in argv if p == PARAMS_FILE]
    assert len(file) == 1, 'only one params.json allowed'
    params = Params(file[0])
    pp.pprint(params)

    create_dirs(argv, params)

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
    params.save(os.path.join(params.run_dir, PARAMS_FILE))
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
