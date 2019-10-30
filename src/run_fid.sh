#!/bin/bash

# $1 log dir
# $2 iteration
# $3 GPU or CPU: --gpu 0, CPU: --gpu -1

python -u ~/git/TTUR/fid.py --gpu $3 ~/src/logs/$1/metrics/fid/$2/images datasets/coco/2017_test/version/v1/fid/te_v1_fid_stats.npz $1 $2 /home/lz01a008/git/TTUR/logs


