import tensorflow as tf


hello = tf.constant('Hello, Tensorflow')

with tf.Session() as sess:
    print(sess.run(hello))

#sess = tf.Session()
#print(sess.run(hello))