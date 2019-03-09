import tensorflow as tf


batch_size = 4

t = tf.constant([[2, 2, 2, 2, 2],[3, 3, 3, 3, 3],[4, 4, 4, 4, 4],[5, 5, 5, 5, 5]], tf.float32)
# t = tf.constant([[4, 9, 16, 25, 30],[5, 10, 17, 26, 31]], tf.float32)
print("t.shape:", t.shape)

d = 3
w = 3

z_b = tf.tile(t, [1, d * w])
print("z_b.shape:", z_b.shape)
matrix = tf.reshape(z_b, [batch_size, d, w, 5])
print("matrix.shape: ", matrix.shape)

x = tf.linspace(tf.constant(-1,tf.float32),tf.constant(1,tf.float32), w)
y = tf.linspace(tf.constant(-1,tf.float32),tf.constant(1,tf.float32), w)

xb,yb = tf.meshgrid(x,y)
print("xb: ", xb)

xb_dim1 = tf.expand_dims(xb, 2)
print("xb_dim1.shape: ", xb_dim1.shape)
# NOT NECESSARY: xb_const = tf.stop_gradient(xb_dim1)
xb_const = xb_dim1
print("xb_const: ", xb_const)

# xb_dim = tf.expand_dims(xb_dim1, 0)
# print("xb_dim.shape: ", xb_dim.shape)
#
# xb_tiled = tf.tile(xb_dim, [5,1,1,1])
# print("xb_tiled.shape: ", xb_tiled.shape)
# print("xb_tiled[0].shape: ", xb_tiled[0].shape)

yb_dim1 = tf.expand_dims(yb, 2)
print("yb_dim1.shape: ", yb_dim1.shape)
# NOT NECESSARY: yb_const = tf.stop_gradient(yb_dim1)
yb_const = yb_dim1
print("yb_const: ", yb_const)

# yb_dim = tf.expand_dims(tf.expand_dims(yb, 2), 0)
# yb_tiled = tf.tile(yb_dim, [5,1,1,1])


def pr(x):
    print('shape:', x.shape)
    print('x: ', x)
    rest = tf.concat(axis=2, values=[x, xb_const, yb_const])
    print('res.shape:', rest.shape)
    return rest

res = tf.map_fn(lambda m: pr(m), matrix)


tf.InteractiveSession()

print("-------------------")
print(res.eval())

# assert 1 == 0
#
# z_sb = tf.concat(axis=3, values=[matrix, xb, yb])
# print("z_sb.shape:", z_sb.shape)
#
#
# # r = tf.ones([25, 4], tf.int32) * t
# # print("r.shape:", r.shape)
# # matrix = tf.reshape(r, [5, 5, 4])
# # print("matrix.shape:", matrix.shape)
#
# tf.InteractiveSession()
#
# # re = z_b.eval()
# #print("re.shape:", re.shape)
# #print(re)
# # re = matrix.eval()
# # print("matrix.shape:", re.shape)
# #print(re)
# print("--------------------------")
# z_sbr = xb.eval()
# print("z_sbr.shape:", z_sbr.shape)
# print(z_sbr)
