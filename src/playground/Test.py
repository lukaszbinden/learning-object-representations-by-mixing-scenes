import tensorflow as tf

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:


	w = tf.get_variable('w', (1536,1),tf.int32)
	v = tf.ones_like(w)

	sess.run(tf.global_variables_initializer())
	vr = sess.run(v)
	r = tf.reshape(v, (512,3))
	
