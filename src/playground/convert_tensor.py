import tensorflow as tf


t = tf.Variable([[4], [9], [16], [25]], tf.int32)
print(t)
ra = tf.rank(t)
print(ra)
r = tf.to_float(t)
print(r)

t2 = tf.ones([10, 64, 64, 1], tf.int32)
print(t2)
r2 = tf.to_float(t2)
print(r2)


x_dim = 64
y_dim = 64
input_tensor = (2,64,64,3)
batch_size_tensor = input_tensor[0]
xx_ones = tf.ones([batch_size_tensor, x_dim], dtype=tf.int32)
xx_ones = tf.expand_dims(xx_ones, -1)
xx_range = tf.tile(tf.expand_dims(tf.range(y_dim), 0), [batch_size_tensor, 1])
xx_range = tf.expand_dims(xx_range, 1)
xx_channel = tf.matmul(xx_ones, xx_range)
xx_channel = tf.expand_dims(xx_channel, -1)
xx_channel = tf.to_float(xx_channel) / (x_dim - 1)
print(xx_channel)

tf.InteractiveSession()

print(ra.eval())
print(xx_channel.eval())