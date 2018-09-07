import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tempfile import TemporaryFile
from matplotlib import pyplot as plt
# learning_rate = 0.08
ratio = 1
img_size = int(28/ratio)

# Reset graph, recreate placeholders and dataset.
def binarize(images, threshold=0.1):
  return (threshold < images).astype('float32')

def get_data():
  return input_data.read_data_sets("MNIST_data/", one_hot=True)

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



class GRUCell:

  def __init__(self, input_dimensions, hidden_size, mode='train', dtype=tf.float32):
    tf.reset_default_graph()
    initializer = tf.contrib.layers.xavier_initializer()
    self.input_dimensions = input_dimensions
    self.hidden_size = hidden_size
    self.mode = mode
    
    self.x, self.y_ = get_placeholders() # feed_dict
    
    W_gru_size = (input_dimensions + hidden_size) * (3 * hidden_size)
    W_fc_size = (hidden_size * 10)
    W_full_size = W_gru_size + W_fc_size
     
    b_gru_size = 3 * hidden_size
    b_fc_size = 10
    b_full_size = b_gru_size + b_fc_size  
    
    if self.mode =='train':
      self.W_full = tf.Variable(initializer([W_full_size]))
      self.b_full = tf.Variable(initializer([b_full_size]))
      
    elif self.mode == 'error':
      self.W_full = tf.placeholder(tf.float32, [W_full_size]) # feed_dict
      self.b_full = tf.placeholder(tf.float32, [b_full_size]) # feed_dict
    
    W_gru_flat = self.W_full[0:W_gru_size]
    W_fc_flat = self.W_full[W_gru_size:]
    self.W_gru = tf.reshape(W_gru_flat, [input_dimensions + hidden_size, 3 * hidden_size])
    self.W_fc = tf.reshape(W_fc_flat, [hidden_size, 10])
    
    # Weights for input vectors of shape (input_dimensions, hidden_size)
    self.Wr = self.W_gru[0:input_dimensions, 0:hidden_size]
    self.Wz = self.W_gru[0:input_dimensions, hidden_size: 2*hidden_size]
    self.Wh = self.W_gru[0:input_dimensions, 2*hidden_size :]
    
    # Weights for hidden vectors of shape (hidden_size, hidden_size)
    self.Ur = self.W_gru[input_dimensions :, 0:hidden_size]
    self.Uz = self.W_gru[input_dimensions :, hidden_size: 2*hidden_size]
    self.Uh = self.W_gru[input_dimensions :, 2*hidden_size:]
    
    # Biases for hidden vectors of shape (hidden_size,)
    self.br = tf.reshape(self.b_full[0:hidden_size], [hidden_size,])
    self.bz = tf.reshape(self.b_full[hidden_size : 2*hidden_size], [hidden_size,])
    self.bh = tf.reshape(self.b_full[2*hidden_size : 3*hidden_size], [hidden_size,])
                       
    self.b_fc = tf.reshape(self.b_full[3*hidden_size:], [10,])
   
    # Define the input layer placeholder
  
    self.input_layer = tf.reshape(self.x, [-1, input_dimensions, input_dimensions])

    # Put the time-dimension upfront for the scan operator
    self.x_t = tf.transpose(self.input_layer, [1, 0, 2], name='x_t')

    # A little hack (to obtain the same shape as the input matrix) to define the initial hidden state h_0
    self.h_0 = tf.matmul(self.x_t[0, :, :], tf.zeros(dtype=tf.float32, shape=(input_dimensions, hidden_size)), name='h_0')

    # Perform the scan operator
    self.h_t_transposed = tf.scan(self.forward_pass, self.x_t, initializer=self.h_0, name='h_t_transposed')

    # Transpose the result back
    self.h_t = tf.transpose(self.h_t_transposed, [1, 0, 2], name='h_t')
    
    self.rnn_output = tf.unstack(self.h_t, self.input_dimensions, 1)[-1]
    
    self.y = tf.matmul(self.rnn_output, self.W_fc) + self.b_fc
    
    self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y_, logits = self.y))
  
  def forward_pass(self, h_tm1, x_t):
    """Perform a forward pass.

    Arguments
    ---------
    h_tm1: np.matrix
        The hidden state at the previous timestep (h_{t-1}).
    x_t: np.matrix
        The input vector.
    """
    # Definitions of z_t and r_t
    z_t = tf.sigmoid(tf.matmul(x_t, self.Wz) + tf.matmul(h_tm1, self.Uz) + self.bz)
    r_t = tf.sigmoid(tf.matmul(x_t, self.Wr) + tf.matmul(h_tm1, self.Ur) + self.br)

    # Definition of h~_t
    h_proposal = tf.tanh(tf.matmul(x_t, self.Wh) + tf.matmul(tf.multiply(r_t, h_tm1), self.Uh) + self.bh)

    # Compute the next hidden state
    h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)

    return h_t
  
  def train_network(self, batch_size=200, num_epochs=150):
    
    mnist = get_data()
    num_epochs = num_epochs
    opt = tf.train.AdamOptimizer()
    train_step = opt.minimize(self.cross_entropy)
    batch_size = batch_size
    
    num_steps = int(num_epochs * 55000 / batch_size)
    
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.y),1), tf.argmax(self.y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    dW = tf.gradients(self.cross_entropy, self.W_full)
    fisher_step = tf.matmul(dW, dW, transpose_a=True)     
    
    i = 0 

    with tf.train.MonitoredSession() as sess:

      while mnist.train.epochs_completed < num_epochs:

        i += 1
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        sess.run(train_step, feed_dict = {self.x: compress_mnist(binarize(batch_xs)), self.y_: batch_ys})

        if i % (1000) == 0:

          test_acc = accuracy.eval(session = sess, feed_dict= 
                          {self.x: compress_mnist(binarize(mnist.test.images)), self.y_: mnist.test.labels})
          loss = sess.run(self.cross_entropy, 
                          feed_dict = {self.x: compress_mnist(binarize(batch_xs)), self.y_: batch_ys})
          print('%d th test acc: %s loss: %s' %(i, test_acc, loss))

        if i == num_steps:

          test_acc = accuracy.eval(session = sess, feed_dict= \
                          {self.x: compress_mnist(binarize(mnist.test.images)), self.y_: mnist.test.labels})
          loss = sess.run(self.cross_entropy, \
                          feed_dict = {self.x: compress_mnist(binarize(mnist.train.images)), self.y_: mnist.train.labels})
          print('FINAL ==== %d th test acc: %s loss: %s' %(i, test_acc, loss))
          self.W_val, self.b_val, self.F = sess.run([self.W_full, self.b_full, fisher_step], feed_dict = {self.x: compress_mnist(binarize(mnist.train.images)), 
                                                                           self.y_: mnist.train.labels})
          
    return self.W_val, self.b_val, self.F
          
          
GRU = GRUCell(28,15,mode='train')
W_val, b_val, F = GRU.train_network()

eig_val_fisher, eig_vec_fisher = np.linalg.eigh(F)

def clip(eig_val):
  return np.log(eig_val[:2085])

fig, ax = plt.subplots()
ax.scatter(np.arange(0, len(clip(eig_val_fisher))), clip(eig_val_fisher), c='blue', label='log eigenvalues')
# ax.scatter(np.arange(0, len(clip(eig_val_h))), clip(eig_val_h), c='red', label='HESSIAN')
ax.legend()
ax.grid(True)
plt.xlabel('dimension')
plt.ylabel('log_eigenvalue')
plt.title('log eigenvalues of Fisher Matrix')
plt.show()

def get_GRU_error(vec):
  GRU = GRUCell(28,15,mode='error')
  with tf.train.MonitoredSession() as sess:
    loss = sess.run(GRU.cross_entropy, feed_dict={
                                             GRU.x: compress_mnist(binarize(mnist.train.images)),
                                             GRU.y_: mnist.train.labels,
                                             GRU.W_full: vec,
                                             GRU.b_full: b_val})
  return loss


def get_quadratic_error(eps, eig_value):
  loss_at_mode = get_GRU_error(W_val)
  second_order_term = (eps**2/2) * (eig_value)
  return loss_at_mode + second_order_term
  
def plot_error(eig_val, eig_vec, which=2084):
  
  which_vec = eig_vec[:, which]
  eig_value = eig_val[which]
  eps_array = np.linspace(-0.5, 0.5 ,num=25)
  error_list = []
  error_quadratic_list = []
  for eps in eps_array:
    dynamic_vec = W_val + eps * which_vec
    error_list.append(get_GRU_error(dynamic_vec))
    error_quadratic_list.append(get_quadratic_error(eps=eps, eig_value=eig_value))
    
  fig, ax = plt.subplots()
  ax.scatter(eps_array, error_list, c='blue', label='true_loss_at_eigendirection')
  ax.scatter(eps_array, error_quadratic_list, c='red', label='quadratic_approximation')
  ax.legend()
  ax.grid(True)
  plt.title('rnn_' + str(which+1)+ '_sub_eigval = ' + str(eig_value))
  plt.xlabel('distance_along_' + str(which+1) + '_eigenvector')
  plt.ylabel('error')
#   plt.savefig('rnn_' + str(which+1)+ '_sub_comparison.png')
#   plt.close()
  
which_list = [2084,
              2083,
              2081,
              2079,
              499,
              2,
              1,
              0]

for which in which_list:
  plot_error(eig_val_fisher, eig_vec_fisher, which)
  
