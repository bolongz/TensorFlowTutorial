import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name ='W')
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name = 'b')
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#define placeholder

xs = tf.placeholder(tf.float32, [None, 784]) #28 * 28
ys = tf.placeholder(tf.float2, [None, 10]) #output

#add_layer

predicton = add_layer(xs, 784, 10, activation_fuction = tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(prediction), reduction_indices = [1])) #loss

train_step = tf.trainGradientDescentOptimizer(-.5).minimize(cross_entrypy)

sess = tf.Session()

sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs, batch_ys = minist.train.next_batch(100)
    sess.run(train_step, feed_dict = {xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
