import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(-1)

fid = 15.3232 * 2
in_s = 5.024212 * 2
fid_sc = tf.constant(fid)
in_s_sc = tf.constant(in_s)

tf.summary.scalar(name='FID', tensor=fid_sc)
tf.summary.scalar(name='IS', tensor=in_s_sc)
summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()

# launch the graph in a session
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs')
    sess.run(init)
    summary = sess.run(summary_op)
    writer.add_summary(summary, 2)
    print('Done with writing the scalar summary')