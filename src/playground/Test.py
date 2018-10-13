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
	# m = np.array([1,1,1,1,1,1,1,1,1])
	# mask = tf.constant(m, tf.float32)
	# p = np.bitwise_xor(m, [1,1,1,1,1,1,1,1,1])
	# prediction = tf.constant(p, tf.float32)
	# res = binary_cross_entropy_with_logits(prediction,mask)
	# res = sess.run(res)
	# print('res: %s' % str(res))


	# mask = tfd.Bernoulli(0.6).sample(9)
	# mask = sess.run(mask)
	# print(mask.shape)
	# print(mask)
	mask = np.array([0,1,1,1,1,1,1,1,0])

	mask = tfd.Bernoulli(0.6).sample(9)
	mask = sess.run(mask)
	print(mask)
	print(m(mask, 1, 0))
	print(m(mask, 1, 1))
	print(m(mask, 2, 0))
	print(m(mask, 2, 1))
	print(m(mask, 9, 0))
	print(m(mask, 9, 1))
