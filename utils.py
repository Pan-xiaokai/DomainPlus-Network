import tensorflow as tf
import numpy as np

def get_session():
   config = tf.ConfigProto()
   config.gpu_options.allow_growth = True
   return tf.Session(config = config)

def data_augmentation(x, method):
   if method == 0:
      return np.rot90(x)
   if method == 1:
      return np.fliplr(x)
   if method == 2:
      return np.flipud(x)
   if method == 3:
      return np.rot90(np.rot90(x))
   if method == 4:
      return np.rot90(np.fliplr(x))
   if method == 5:
      return np.rot90(np.flipud(x))
