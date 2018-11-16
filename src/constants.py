import tensorflow as tf

JSON_FILE_PARAM = '-p='
JSON_FILE_DEFAULT = 'params.json'
COMMENT_PARAM = '-c='
LOG_FILE_NAME = 'main.log'

NUM_TILES = 9
NUM_TILES_L2_MIX = 4
FROM_X1 = 1
FROM_I1 = FROM_X1
FROM_X2 = 0
FROM_I2 = FROM_X2
NUM_CROPS = 3
FROM_I_REF = 0
FROM_I_M = 1

SPECTRAL_NORM_UPDATE_OPS = "spectral_norm_update_ops"

underscore = tf.constant("_")
z1 = tf.constant("0")
z2 = tf.constant("00")
z3 = tf.constant("000")
z4 = tf.constant("0000")
z5 = tf.constant("00000")
z6 = tf.constant("000000")
z7 = tf.constant("0000000")
z8 = tf.constant("00000000")
z9 = tf.constant("000000000")
z10 = tf.constant("0000000000")
z11 = tf.constant("00000000000")