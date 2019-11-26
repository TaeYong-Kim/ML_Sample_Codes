import tensorflow as tf

#When we knew Input data 
'''
x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random.normal([1]), name="weight")
b = tf.Variable(tf.random.normal([1]), name="bias")

hypothesis = x_train * W + b
cost = tf.reduce_mean (tf.square(hypothesis - y_train))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if(step % 20 == 0):
        print(step, '\t', sess.run(cost), '\t', sess.run(W), '\t', sess.run(b))
'''

#Get random input (use placeholder)
x = tf.placeholder(tf.float32, shape = [None])
y = tf.placeholder(tf.float32, shape = [None])


W = tf.Variable(tf.random.normal([1]), name="weight")
b = tf.Variable(tf.random.normal([1]), name="bias")

hypothesis = x * W + b;
cost = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    _cost, _W, _b, _ = sess.run([cost, W, b, train], feed_dict={x:[1,2,3,4,5], y:[2.1,4.1,6.1,8.1,10.1]})

    if step % 20 == 0:
        print(step, _cost, _W, _b)