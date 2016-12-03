import input_data
import random

import tensorflow as tf

dataset = input_data.getImagesWithLabels()
random.shuffle(dataset)

images = []
labels = []

for data in dataset:
    images.append(data[3])
    labels.append(data[4])

# Set parameters
learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2

imgbytes = 3 * 100 * 100
numlabels = 10 + 26 + 26

# TF graph input
x = tf.placeholder("float", [None, imgbytes])  # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, numlabels])  # 0-9 digits recognition => 10 classes

# Create a model

# Set model weights
W = tf.Variable(tf.zeros([imgbytes, numlabels]))
b = tf.Variable(tf.zeros([numlabels]))

with tf.name_scope("Wx_b") as scope:
    # Construct a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

# Add summary ops to collect data
w_h = tf.histogram_summary("weights", W)
b_h = tf.histogram_summary("biases", b)

# More name scopes will clean up graph representation
with tf.name_scope("cost_function") as scope:
    # Minimize error using cross entropy
    # Cross entropy
    cost_function = -tf.reduce_sum(y * tf.log(model))
    # Create a summary to monitor the cost function
    tf.scalar_summary("cost_function", cost_function)

with tf.name_scope("train") as scope:
    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Initializing the variables
init = tf.initialize_all_variables()

# Merge all summaries into a single operator
merged_summary_op = tf.merge_all_summaries()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Change this to a location on your computer
    summary_writer = tf.train.SummaryWriter('./logs', graph_def=sess.graph_def)

    sess.run(optimizer, feed_dict={x: images[:7500], y: labels[:7500]})

    # Compute the average loss
    sess.run(cost_function, feed_dict={x: images[:7500], y: labels[:7500]})

    # Write logs for each iteration
    summary_str = sess.run(merged_summary_op, feed_dict={x: images[:7500], y: labels[:7500]})

    # Test the model
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuracy:", accuracy.eval({x: images[7500:], y: labels[7500:]}))
