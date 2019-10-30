import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # str(params.gpu)


# LZ 09.03: this test should demonstrate that variables x, y remain unchanged through training/BP as they are
# wrapped by tf.stop_gradient while variable t changes with every iteration...
# IN FACT: IT APPEARS THAT tf.linspace  DOES NOT CREATE A VARIABLE BUT A CONSTANT THAT IS NOT AVAILABLE FOR BP!, SO
# THE USE OF tf.stop_gradient IS UNNECESSARY!

batch_size = 2
s = tf.placeholder(tf.float32, [1])
gt = tf.placeholder(tf.float32,[2,3,3,7])
t = tf.Variable([[4., 9., 16., 25., 30.],[5., 10., 17., 26., 31.]], tf.float32)
print("t.shape:", t.shape)

d = 3
w = 3

z_b = tf.tile(t, [1, d * w])
print("z_b.shape:", z_b.shape)
matrix = tf.reshape(z_b, [batch_size, d, w, 5])
print("matrix.shape: ", matrix.shape)

x = tf.linspace(tf.constant(-1,tf.float32),tf.constant(1,tf.float32), w)
print('x:', x)
y = tf.linspace(tf.constant(-1,tf.float32),tf.constant(1,tf.float32), w)
print('y:', y)

xb,yb = tf.meshgrid(x,y)
print("xb: ", xb)

xb_dim1 = tf.expand_dims(xb, 2)
print("xb_dim1.shape: ", xb_dim1.shape)
#xb_const = tf.stop_gradient(xb_dim1)
xb_const = xb_dim1
print("xb_const: ", xb_const)

yb_dim1 = tf.expand_dims(yb, 2)
print("yb_dim1.shape: ", yb_dim1.shape)
# yb_const = tf.stop_gradient(yb_dim1)
yb_const = yb_dim1
print("yb_const: ", yb_const)

# yb_dim = tf.expand_dims(tf.expand_dims(yb, 2), 0)
# yb_tiled = tf.tile(yb_dim, [5,1,1,1])

# matrix = tf.stop_gradient(matrix)

def pr(u):
    print('shape:', u.shape)
    print('x: ', u)
    rest = tf.concat(axis=2, values=[u, xb_const, yb_const])
    print('res.shape:', rest.shape)
    return rest

res = tf.map_fn(lambda m: pr(m), matrix)

output = res * s
print("output: ", output.shape)

loss = output - gt
optimizer = tf.train.GradientDescentOptimizer(1).minimize(loss)


iter = 3
i = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("*****before gradient descent*****")
    print("t---\n", t.eval(), "\n", "\n\nxb---\n", xb_const.eval(), "\n\nyb---\n", yb_const.eval())
    while i < iter:
        t_,xb_,yb_,_ = sess.run([t,xb_const,yb_const,optimizer],feed_dict = {s:np.random.normal(size = 1),gt:np.random.normal(size = (2,3,3,7))})
        print("*****after gradient descent*****")
        print("\nt---\n", t_, "\n", "\n\nxb---\n", xb_, "\n\nyb---\n", yb_)
        i += 1