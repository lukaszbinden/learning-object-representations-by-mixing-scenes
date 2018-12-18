import tensorflow as tf
from ops_alex import binary_cross_entropy_with_logits
import numpy as np
tfd = tf.contrib.distributions

def m(mk, tile, target):
	# target == 1 corresponds tile from x1
	# target == 0 corresponds tile from x2
	return 1 if mk[tile-1] == target else 0

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
	sess.run(tf.global_variables_initializer())

	batch_size = 24
	feature_size_tile = 192
	feature_size = feature_size_tile * 4
	feature_tile_shape = [batch_size, feature_size_tile]

	I_t1_f = tf.ones((batch_size, feature_size))
	t1_f = tf.slice(I_t1_f, [0, feature_size_tile * 0], feature_tile_shape)

	t1 = tf.ones((batch_size, feature_size_tile))
	t2 = tf.ones((batch_size, feature_size_tile)) * 2
	t3 = tf.ones((batch_size, feature_size_tile)) * 3
	t4 = tf.ones((batch_size, feature_size_tile)) * 4

	concatenated = tf.concat(axis=1, values=[t1, t2, t3, t4])

	assert I_t1_f.shape == concatenated.shape

	r1 = tf.slice(I_t1_f, [0, feature_size_tile * 0], feature_tile_shape)
	r2 = tf.slice(I_t1_f, [0, feature_size_tile * 1], feature_tile_shape)
	assert r1.shape == r2.shape

	v1 = tf.slice(concatenated, [0, feature_size_tile * 0], feature_tile_shape)
	rest = tf.slice(concatenated, [0, feature_size_tile * 1], [batch_size, feature_size_tile * 3])
	concatenated_restored = tf.concat(axis=1, values=[v1, rest])

	v2 = tf.slice(concatenated, [0, feature_size_tile * 1], feature_tile_shape)
	v3 = tf.slice(concatenated, [0, feature_size_tile * 2], feature_tile_shape)
	v4 = tf.slice(concatenated, [0, feature_size_tile * 3], feature_tile_shape)


	r1 = sess.run(r1)
	rest = sess.run(rest)
	concatenated_restored = sess.run(concatenated_restored)
	concatenated = sess.run(concatenated)
	r2 = sess.run(r2)
	v1 = sess.run(v1)
	v2 = sess.run(v2)
	v3 = sess.run(v3)
	v4 = sess.run(v4)

	print(r1.shape)
	print(rest.shape)
	print(concatenated_restored.shape)
	print(concatenated.shape)
	print(r2.shape)
	print(v1.shape)
	print(v2.shape)
	print(v3.shape)
	print(v4.shape)
	print('r1 ----------')
	print(r1)
	print('rest ----------')
	print(rest)
	print('concatenated_restored ----------')
	print(concatenated_restored)
	print('concatenated ----------')
	print(concatenated)
	print(np.array_equal(concatenated_restored, concatenated))
	print('v1 ----------')
	print(v1)
	print('v2 ----------')
	print(v2)
	print('v3 ----------')
	print(v3)
	print('v4 ----------')
	print(v4)

