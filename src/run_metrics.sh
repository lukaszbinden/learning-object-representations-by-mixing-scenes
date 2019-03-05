#!/bin/bash

# $1 = expXY
# $2 = folder e.g. 20190211_165210

nohup python -u metrics_main.py -exp=$1 -test_from=$2 -main=lorbms_main_$1.py -p=params_$1.json > nohup_metrics_$1_$2.out &


