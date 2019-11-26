import tensorflow as tf

x = tf.compat.v1.placeholder(tf.float32, shape = [None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

W = tf.Variable(tf.random.normal([2,1]), name = "weight")
b = tf.Variable(tf.random.normal([1]), name = "bias")

hypothesis = tf.sigmoid(tf.matmul(x,W)+b)

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * (tf.log(1-hypothesis)))

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.01).minimize (cost)

prediction = tf.cast(hypothesis>0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction,y), dtype = tf.float32))

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range (10001):
        cost_val, _ = sess.run([cost, train], feed_dict = {x:x_data,y:y_data})

        if step %1000 == 0:
            print(step, '\t', cost_val)