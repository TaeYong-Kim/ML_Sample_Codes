import tensorflow as tf

hello = tf.constant('Hello, Tensorflow')

#1st way to use Session
with tf.Session() as sess:
    print(sess.run(hello))

#2nd way to use Session
#sess = tf.Session()
#print(sess.run(hello))