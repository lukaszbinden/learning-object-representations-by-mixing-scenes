import tensorflow as tf
from ops_alex import binary_cross_entropy_with_logits
import numpy as np
tfd = tf.contrib.distributions

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
	sess.run(tf.global_variables_initializer())
	m = np.array([1,1,1,1,1,1,1,1,1])
	mask = tf.constant(m, tf.float32)
	p = np.bitwise_xor(m, [1,1,1,1,1,1,1,1,1])
	prediction = tf.constant(p, tf.float32)
	res = binary_cross_entropy_with_logits(prediction,mask)
	res = sess.run(res)
	print('res: %s' % str(res))
