import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class SubMnist():

  def __init__(self, num_examples, num_epochs):
    self._num_epochs = num_epochs
    self._num_examples = num_examples
    self._index_in_epoch = 0
    self._epochs_completed = 0
    self.get_data()

  def get_data(self):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    sub_mnist = mnist.train.next_batch(self._num_examples)
    self.sub_images = sub_mnist[0]
    self.sub_labels = sub_mnist[1]

  def next_batch(self,batch_size,shuffle = True):
    start = self._index_in_epoch

    if start == 0 and self._epochs_completed == 0:
      idx = np.arange(0, self._num_examples)  # get all possible indexes
      np.random.shuffle(idx)  # shuffle indeces
      self._sub_images = self.sub_images[idx]
      self._sub_labels = self.sub_labels[idx]

    # go to the next batch
    if start + batch_size > self._num_examples:
      self._epochs_completed += 1
      rest_num_examples = self._num_examples - start
      images_rest = self._sub_images[start:self._num_examples]
      labels_rest = self._sub_labels[start:self._num_examples]

      idx0 = np.arange(0, self._num_examples)  # get all possible indexes
      np.random.shuffle(idx0)  # shuffle indexes
      self._sub_images = self.sub_images[idx0]
      self._sub_labels = self.sub_labels[idx0]

      start = 0
      self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
      end =  self._index_in_epoch  
      sub_images_new = self._sub_images[start:end]
      sub_labels_new = self._sub_labels[start:end]
      return np.concatenate((images_rest, sub_images_new), axis=0), np.concatenate((labels_rest, sub_labels_new), axis=0)

    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._sub_images[start:end], self._sub_labels[start:end]
