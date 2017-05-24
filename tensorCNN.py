from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import datetime

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
# summary_writer = tf.train.SummaryWriter('/tmp/logs', sess.graph_def)

#784 inputs for pixels, 10 outputs
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#input to conv kernel weights
W = tf.Variable(tf.zeros([784,10]))
#input to conv layer bias
b = tf.Variable(tf.zeros([10]))

#weights for first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 64])
#bias for first convolutional layer
b_conv1 = bias_variable([64])

#reshape x into a tensor
x_image = tf.reshape(x, [-1,28,28,1])
#take the activation of the convolution between tensor and kernels, plus bias
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 64, 128])
b_conv2 = bias_variable([128])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 128, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

training_start = datetime.datetime.now()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('tmp/logs', sess.graph)
sess.run(tf.global_variables_initializer())

for i in range(2000):
    print("step %d"%i)
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    summary_str, _ = sess.run([summary,accuracy], feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    summary_writer.add_summary(summary_str, i)

training_end = datetime.datetime.now()
training_time = training_end - training_start
datetime.timedelta(0, 125, 749430)
time = divmod(training_time.total_seconds(), 60)
print("### TensorFlow Multi-Layer CNN ###")
print("training time: %d minutes and %f seconds"%time)
testing_start = datetime.datetime.now()
accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
testing_end = datetime.datetime.now()
testing_time = testing_end - testing_start
time = divmod(testing_time.total_seconds(), 60)
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
print("testing time: %d minutes and %f seconds"%time)

saver = tf.train.Saver()
saver.save(sess, 'model/')
