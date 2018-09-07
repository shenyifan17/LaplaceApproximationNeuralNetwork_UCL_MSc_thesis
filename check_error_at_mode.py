# test the impact on eigen-spectrum of a simple fully connected neural network with a small amount of training data
# save: 
# images of the log eigen spectrum, 
# largest eigenvalues againts training size, 
# smallest eigenvalues against training size, 
# how sparse the eig_val against training size

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from matplotlib import pyplot as plt
from SubMnist import SubMnist as Sub

# Import dataset with one-hot encoding of the class labels.
def get_data():
  return input_data.read_data_sets("MNIST_data/", one_hot=True)

mnist = get_data()
eval_mnist = get_data()

def get_placeholders(img_size):
  x = tf.placeholder(tf.float32, [None, img_size ** 2])
  y_ = tf.placeholder(tf.float32, [None, 10])
  return x, y_

def compress_mnist(batch_xs, ratio=2):
  batch_size = np.shape(batch_xs)[0]
  mnist_reshaped = batch_xs.reshape(batch_size, 28, 28)
  mnist_compressed = mnist_reshaped[:, ::ratio, ::ratio]
  return mnist_compressed.reshape(batch_size,-1)

def plt_eig_val_on_dif_training_size(sub_sample_size=500, 
                                     num_epochs=2000, 
                                     learning_rate=0.05, 
                                     batch_size=200,
                                     log_period_samples=50000,
                                     ratio=2,
                                     num_hidden_units=25):
  
  num_steps = int(num_epochs * sub_sample_size / batch_size)
  print('number_of_steps is ====== ', num_steps )

  img_size = int(28/ratio)

  # initialise a sub mnist class, by choosing a subset of training data
  SubMnist = Sub(num_examples=sub_sample_size, num_epochs=num_epochs)

  # Placeholders to feed train and test data into the graph.
  # Since batch dimension is 'None', we can reuse them both for train and eval.

  x, y_ = get_placeholders(img_size=img_size)

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
      sess.run(train_step, feed_dict={x: compress_mnist(batch_xs, ratio=ratio)
                                      ,y_:batch_ys})
      #################

      if i == num_steps: # when training finished
        print('............... training completed, saving variables ..............')
        w_star, hessian, b_1_val, b_2_val = sess.run([w_full_flat, hessian_step, b_1, b_2], feed_dict={x: compress_mnist(batch_xs, ratio=ratio), y_:batch_ys})

  print('===================== number of training samples ====================', sub_sample_size)
  print('===================== number of parameters ====================', hessian[0].shape[0])

  eig_val, eig_vec = np.linalg.eigh(hessian[0])
  how_sparse = np.sum((np.abs(eig_val) < 1e-5) * 1)/len(eig_val)
  print('how_sparse', how_sparse)
  threshold = 1e-5
  log_eig_val = np.log(eig_val[eig_val>threshold])
  # eigenvalue spectrum
  plt.scatter(np.arange(0, len(log_eig_val)), log_eig_val)
  plt.xlabel('dimension')
  plt.ylabel('log_scale_eigenvalue')
  plt.title('log_eig_val sample: ' + str(sub_sample_size) + '; threshold: ' + str(threshold) + '; sparse: ' + str(int(how_sparse * 100)) + '%')
  plt.grid()
  plt.savefig('log_eig_val_'+str(sub_sample_size)+'.png')
  plt.close()
  return eig_val, how_sparse

# body: 
sample_size_list = [200, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
result_store = {}
eig_large_list = []
eig_small_list = []
sparse_list = []

for sz in sample_size_list:
	eig_val_large_avg = 0
	eig_val_small_avg = 0
	sparse_avg = 0
	num = 20
	for i in range(num):
		eig_val, how_sparse = plt_eig_val_on_dif_training_size(sub_sample_size=sz)
		eig_val_large_avg += eig_val.max()/num
		eig_val_small_avg += eig_val.min()/num
		sparse_avg += how_sparse/num

	eig_large_list.append(eig_val_large_avg)
	eig_small_list.append(eig_val_small_avg)
	sparse_list.append(sparse_avg)

plt.scatter(sample_size_list, eig_large_list)
plt.plot(sample_size_list, eig_large_list)
plt.title('largest eigenvalues against training size')
plt.xlabel('number_of_training_data')
plt.ylabel('largest_eigenvalues')
plt.grid()
plt.savefig('large_eig.png')
plt.close()

plt.scatter(sample_size_list, eig_small_list)
plt.plot(sample_size_list, eig_small_list)
plt.title('smallest eigenvalues against training size')
plt.xlabel('number_of_training_data')
plt.ylabel('smallest_eigenvalues')
plt.grid()
plt.savefig('small_eig.png')
plt.close()

plt.scatter(sample_size_list, sparse_list)
plt.plot(sample_size_list, sparse_list)
plt.title('how sparse against training size')
plt.xlabel('number_of_training_data')
plt.ylabel('percentage_of_small_eigenvalues')
plt.grid()
plt.savefig('sparse.png')
plt.close()

