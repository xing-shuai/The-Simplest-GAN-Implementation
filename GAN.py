import tensorflow as tf
import numpy as np

x_input = tf.placeholder(tf.float32, [None, 1])

y_real = tf.placeholder(tf.float32, [None, 1])

with tf.variable_scope("G"):
    g_w = tf.Variable([0.1], dtype=tf.float32)
    g_b = tf.Variable([0.0], dtype=tf.float32)
    g_out = x_input * g_w + g_b

with tf.variable_scope("D"):
    # d_w = tf.Variable([0.1], dtype=tf.float32)
    # d_b = tf.Variable([0.0], dtype=tf.float32)
    # prob_0 = tf.nn.sigmoid(g_out * d_w + d_b)
    # prob_1 = tf.nn.sigmoid(y_real * d_w + d_b)

    D_l0 = tf.layers.dense(y_real, 12, tf.nn.relu, name='D_l')
    prob_1 = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name='D_out')

    D_l1 = tf.layers.dense(g_out, 12, tf.nn.relu, name='D_l', reuse=True)
    prob_0 = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='D_out', reuse=True)

g_loss = tf.reduce_mean(tf.log(1 - prob_0))
d_loss = -tf.reduce_mean(tf.log(prob_1) + tf.log(1 - prob_0))

g_train = tf.train.GradientDescentOptimizer(0.01).minimize(g_loss, var_list=[g_w, g_b])
# d_train = tf.train.GradientDescentOptimizer(0.01).minimize(d_loss, var_list=[d_w, d_b])
d_train = tf.train.GradientDescentOptimizer(0.1).minimize(d_loss,
                                                          var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                     scope='D'))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        g_parameter = sess.run([g_w, g_b])
        print("g_w %f   g_b %f " % (g_parameter[0][0], g_parameter[1][0]))
        sess.run([d_train, g_train], feed_dict={x_input: np.random.randn(10).reshape([10, 1]),
                                                y_real: np.random.randn(10).reshape([10, 1]) * 1.6 + 0.8})

        # if i % 100 == 0:
        # break
