num_epochs = 3000
learning_rate = 0.05
sub_sample_size = 2000

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from matplotlib import pyplot as plt
from SubMnist import SubMnist as Sub

# Reset graph, recreate placeholders and dataset.

batch_size = 200
log_period_samples = 50000

num_steps = int(num_epochs * sub_sample_size / batch_size)
print('number_of_steps is ====== ', num_steps )

ratio = 2
img_size = int(28/ratio)

num_hidden_units = 25
# Import dataset with one-hot encoding of the class labels.
def get_data():
  return input_data.read_data_sets("MNIST_data/", one_hot=True)
    
SubMnist = Sub(num_examples=sub_sample_size, num_epochs=num_epochs)

# Placeholders to feed train and test data into the graph.
# Since batch dimension is 'None', we can reuse them both for train and eval.
def get_placeholders():
  x = tf.placeholder(tf.float32, [None, img_size ** 2])
  y_ = tf.placeholder(tf.float32, [None, 10])
  return x, y_

def compress_mnist(batch_xs, ratio=ratio):
  batch_size = np.shape(batch_xs)[0]
  mnist_reshaped = batch_xs.reshape(batch_size, 28, 28)
  mnist_compressed = mnist_reshaped[:, ::ratio, ::ratio]
  return mnist_compressed.reshape(batch_size,-1)
   
x, y_ = get_placeholders()
mnist = get_data()
eval_mnist = get_data()
#####################################################
# Define model, loss, update and evaluation metric. #
# Model   
initializer = tf.contrib.layers.xavier_initializer()

num_w1_units = (img_size ** 2) * num_hidden_units
num_w2_units = num_hidden_units * 10

w_full_flat = tf.Variable(initializer([num_w1_units + num_w2_units]))
W_1_flat = w_full_flat[0:num_w1_units]
W_1 = tf.reshape(W_1_flat, [img_size ** 2, num_hidden_units])
b_1 = tf.Variable(initializer([num_hidden_units]))
hid_1 = tf.nn.relu(tf.matmul(x,W_1) + b_1)
# Model (2nd linear layer)
W_2_flat = w_full_flat[num_w1_units : num_w1_units + num_w2_units]
W_2 = tf.reshape(W_2_flat, [num_hidden_units, 10])
b_2 = tf.Variable(initializer([10]))
y = tf.matmul(hid_1,W_2) + b_2

# loss
cross_entropy = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# update 
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
hessian_step = tf.hessians(cross_entropy, w_full_flat)

# evaluation metric for train
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y),1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#####################################################

# Train.
i, train_accuracy, test_accuracy = 0, [], []
log_period_updates = 2000
with tf.train.MonitoredSession() as sess:
  tf.reset_default_graph()
  while SubMnist._epochs_completed < num_epochs:

    # Update.
    i += 1
    batch_xs, batch_ys = SubMnist.next_batch(batch_size)
    # We use test datasets here, so that we have more parameters than training samples    

    #################
    # Training step #
    sess.run(train_step, feed_dict={x: compress_mnist(batch_xs)
                                    ,y_:batch_ys})
    #################

    # Periodically evaluate.
    if i % log_period_updates == 0:
      #####################################
      # Compute and store train accuracy. #
      train_acc = accuracy.eval(session = sess, feed_dict={x: compress_mnist(SubMnist.sub_images), y_: SubMnist.sub_labels})
      train_accuracy.append(train_acc)
      #####################################

      #####################################
      # Compute and store test accuracy.  #
      test_acc = accuracy.eval(session = sess, feed_dict={x: compress_mnist(eval_mnist.train.images), y_: eval_mnist.train.labels})
      test_accuracy.append(test_acc)
      #######################¢##############
      print('%d th training accuracy: %s test accuracy: %s' %(i,train_acc,test_acc))

    if i == num_steps: # when training finished
      print('............... training completed, saving variables ..............')
      w_star, hessian, b_1_val, b_2_val = sess.run([w_full_flat, hessian_step, b_1, b_2], feed_dict={x: compress_mnist(batch_xs),y_:batch_ys})

print('===================== number of training samples ====================', sub_sample_size)
print('===================== number of parameters ====================', hessian[0].shape[0])

eig_val, eig_vec = np.linalg.eigh(hessian[0])
threshold = 1e-5
log_eig_val = np.log10(eig_val[eig_val>threshold])
plt.scatter(np.arange(0, len(log_eig_val)), log_eig_val)
plt.plot(log_eig_val)
plt.grid(True)
plt.title('log_scale_eig_spectrum_sub_data_fc')
plt.xlabel('dimension')
plt.ylabel('log_eigenvalue')
plt.savefig('eig_spectrum_sub_data_fc.png')
plt.close()

# plt.plot(train_accuracy, c='blue', label='train_accuracy')
# plt.plot(test_accuracy, c='red', label='test_accuracy')
# plt.grid(True)
# plt.xlabel('number_of_steps')
# plt.ylabel('accuracy')
# plt.title('learning_curve_sub_data')
# plt.savefig('learning_curve_sub_data.png')
# plt.close()

# analyse results

def get_w_placeholders():
  w_full_flat = tf.placeholder(tf.float32, [num_w1_units + num_w2_units])
  b_1 = tf.placeholder(tf.float32, [num_hidden_units])
  b_2 = tf.placeholder(tf.float32, [10])
  return w_full_flat, b_1, b_2

# build a new graph for calculating the error:
def get_error_with_train(vec):

  x, y_ = get_placeholders()
  w_full_flat, b_1_pl, b_2_pl = get_w_placeholders()
  

  W_1_flat = w_full_flat[0:num_w1_units]
  W_1 = tf.reshape(W_1_flat, [img_size ** 2, num_hidden_units])
  hid_1 = tf.nn.relu(tf.matmul(x,W_1) + b_1_pl)
  # Model (2nd linear layer)
  W_2_flat = w_full_flat[num_w1_units : num_w1_units + num_w2_units]
  W_2 = tf.reshape(W_2_flat, [num_hidden_units, 10])
  y = tf.matmul(hid_1,W_2) + b_2_pl

  # loss
  cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

  with tf.train.MonitoredSession() as sess:
    tf.reset_default_graph()
    cross_entropy_error = sess.run(cross_entropy, 
                         feed_dict={x: compress_mnist(mnist.train.images), 
                                    y_: mnist.train.labels,
                                    w_full_flat: vec,
                                    b_1_pl: b_1_val,
                                    b_2_pl: b_2_val})
    return cross_entropy_error

  
def get_quadratic_error(eps, eig_value):
  # quadratic approximation at posterior mode
  loss_at_mode = get_error_with_train(w_star)
  second_order_term = (eps**2/2) * eig_value
  return loss_at_mode + second_order_term

def plot_error(mode='eig_large', which=5000):
  if mode == 'eig_large':
    which = eig_val.argmax()
    which_vec = eig_vec[:, which]
    eig_value = eig_val.max()
  elif mode == 'eig_choose':
    # choose a random eig vec
    print(which)
    which_vec = eig_vec[:, which]
    eig_value = eig_val[which]
  eps_array = np.linspace(-1, 1 ,num=50)
  error_list = []
  error_quadratic_list = []
  # mnist = get_data()
  for eps in eps_array:
    dynamic_vec = w_star + eps * which_vec
    error_list.append(get_error_with_train(dynamic_vec))
    error_quadratic_list.append(get_quadratic_error(eps=eps, eig_value=eig_value))
 
  fig, ax = plt.subplots()
  ax.scatter(eps_array, error_list, c='blue', label='true_loss_at_eigendirection')
  ax.scatter(eps_array, error_quadratic_list, c='red', label='quadratic_approximation')
  ax.legend()
  ax.grid(True)
  plt.title(str(which+1)+ '_sub_eigval = ' + str(eig_value))
  plt.xlabel('distance_along_' + str(which+1) + '_eigenvector')
  plt.ylabel('error')
  plt.savefig('fc_' + str(which+1)+ '_sub_comparison.png')
  plt.close()

# body 
which_list = [5149,
              5148,
              5147, 
              5146,
              5145,
              5144,
              5143, 
              5142,
              5141,
              5140,
              5139,
              5138,
              5137,
              5136,
              5135, 
              5134, 
              5133,
              5132,
              5131,
              5130,
              5129,
              5124,
              5119,
              5114,
              5109,
              5104, 
              5099, 
              5074,
              5049,
              5024, 
              4999, 
              4799,
              4499,
              4249,
              3999,
              3499,
              2999,
              2499,
              1999,
              1499,
              999,
              749,
              499,
              299,
              199,
              149,
              99,
              49,
              24,
              14,
              9,
              4,
              3,
              2,
              1,
              0]


# plot_error(mode='eig_large')

# for which in which_list:
#   plot_error(mode='eig_choose', which=which)

