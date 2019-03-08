import tensorflow as tf


t = tf.constant([4, 9, 16, 25, 30], tf.float32)
print("t.shape:", t.shape)

d = 4
w = 4

z_b = tf.tile(t, [d * w])
print("r.shape:", z_b.shape)
matrix = tf.reshape(z_b, [d, w, 5])
print("matrix.shape: ", matrix)

x = tf.linspace(tf.constant(-1,tf.float32),tf.constant(1,tf.float32), w)
y = tf.linspace(tf.constant(-1,tf.float32),tf.constant(1,tf.float32), w)

xb,yb = tf.meshgrid(x,y)
print("xb.shape: ", xb.shape)

xb = tf.expand_dims(xb, 2)
print("xb2.shape: ", xb.shape)
yb = tf.expand_dims(yb, 2)

z_sb = tf.concat(axis=-1, values=[matrix, xb, yb])
print("z_sb.shape:", z_b.shape)

# r = tf.ones([25, 4], tf.int32) * t
# print("r.shape:", r.shape)
# matrix = tf.reshape(r, [5, 5, 4])
# print("matrix.shape:", matrix.shape)

tf.InteractiveSession()

# re = z_b.eval()
#print("re.shape:", re.shape)
#print(re)
# re = matrix.eval()
# print("matrix.shape:", re.shape)
#print(re)
print("--------------------------")
z_sbr = z_sb.eval()
print("z_sbr.shape:", z_sbr.shape)
print(z_sbr)
