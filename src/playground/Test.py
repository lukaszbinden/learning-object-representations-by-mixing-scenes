import tensorflow as tf

counter = 4

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	
	gs = tf.Variable(counter, name='global_step', trainable=False)
	print(gs)
	tf.global_variables_initializer().run()
	
	counter = 33
	# gs = tf.assign(gs, 12)
	#gs.load(counter)
	print(gs)
	print(sess.run(gs))
	
	