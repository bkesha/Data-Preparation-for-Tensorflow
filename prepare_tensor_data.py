import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import urllib
import numpy as np
import zipfile
import os
import os.path  
import glob  
from skimage.transform import resize


def imcrop_tosquare(img):
    
    size = np.min(img.shape[:2])
    extra = img.shape[:2] - size
    crop = img
    for i in np.flatnonzero(extra):
        crop = np.take(crop, extra[i] // 2 + np.r_[:size], axis=i)
    return crop

def gabor(ksize=32):
    
    g = tf.Graph()
    with tf.Session(graph=g):
        z_2d = gauss2d(0.0, 1.0, ksize)
        ones = tf.ones((1, ksize))
        ys = tf.sin(tf.linspace(-3.0, 3.0, ksize))
        ys = tf.reshape(ys, [ksize, 1])
        wave = tf.matmul(ys, ones)
        gabor = tf.mul(wave, z_2d)
        return gabor.eval()


def convolve(img, kernel):
    
    g = tf.Graph()
    with tf.Session(graph=g):
        convolved = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')
        res = convolved.eval()
    return res

def gauss2d(mean, stddev, ksize):
    
    z = gauss(mean, stddev, ksize)
    g = tf.Graph()
    with tf.Session(graph=g):
        z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))
        return z_2d.eval()
def gauss(mean, stddev, ksize):
    
    g = tf.Graph()
    with tf.Session(graph=g):
        x = tf.linspace(-3.0, 3.0, ksize)
        z = (tf.exp(tf.neg(tf.pow(x - mean, 2.0) /
                           (2.0 * tf.pow(stddev, 2.0)))) *
             (1.0 / (stddev * tf.sqrt(2.0 * 3.1415))))
        return z.eval()


def prepare(dir,size):
  dirname = dir
  size = size
  filenames = [os.path.join(dirname, fname) for fname in os.listdir(dirname)]
  filenames = filenames[:size]

  # Read every filename as an RGB image
  imgs = [plt.imread(fname)[...,:3] for fname in filenames]

  # Crop every image to a square
  imgs = [imcrop_tosquare(img_i) for img_i in imgs]

  #Then resize the square image to 100 x 100 pixels
  imgs = [resize(img_i, (size, size)) for img_i in imgs]

  # Finally make our list of 3-D images a 4-D array with the first dimension the number of images:
  imgs = np.array(imgs).astype(np.float32)

  #imgs = np.asarray([transform.resize(im, SIZE) for im in circle])

  # First create a tensorflow session
  sess = tf.Session()

  # Now create an operation that will calculate the mean of your images
  mean_img_op = tf.reduce_mean(imgs,0)

  # And then run that operation using your session
  mean_img = sess.run(mean_img_op)

  # Create a tensorflow operation to give you the standard deviation

  # First compute the difference of every image with a
  # 4 dimensional mean image shaped 1 x H x W x C
  mean_img_4d = tf.reduce_mean(imgs, reduction_indices=0, keep_dims=True)

  subtraction = imgs - mean_img_4d

  # Now compute the standard deviation by calculating the
  # square root of the expected squared differences
  std_img_op = tf.sqrt(tf.reduce_mean(subtraction * subtraction, reduction_indices=0))

  # Now calculate the standard deviation using your session
  std_img = sess.run(std_img_op)

  norm_imgs_op = (tf.exp(tf.neg(tf.pow(imgs - mean_img_4d, 2.0) /
                     (2.0 * tf.pow(std_img, 2.0)))) *
       (1.0 / (std_img * tf.sqrt(2.0 * 3.1415))))

  norm_imgs = sess.run(norm_imgs_op)

  norm_imgs_show = (norm_imgs - np.min(norm_imgs)) / (np.max(norm_imgs) - np.min(norm_imgs))


  # First build 3 kernels for each input color channel
  ksize = ksize = norm_imgs_op.get_shape().as_list()[0]
  kernel = np.concatenate([gabor(ksize)[:, :, np.newaxis] for i in range(3)], axis=2)
                         
  # Now make the kernels into the shape: [ksize, ksize, 3, 1]:
  kernel_4d = kernel.reshape([ksize,ksize,3,1])
  #assert(kernel_4d.shape == (ksize, ksize, 3, 1))


  convolved = convolve(imgs,kernel_4d)

  convolved_show = (convolved - np.min(convolved)) / (np.max(convolved) - np.min(convolved))


  
  flattened = tf.reshape(convolved,[size,size*size])
  
  # Now calculate some statistics about each of our images
  values = tf.reduce_sum(flattened, reduction_indices=1)

  # Then create another operation which sorts those values
  # and then calaculate the result:
  idxs_op = tf.nn.top_k(values, k=size)[1]
  idxs = sess.run(idxs_op)

  # Then finally use the sorted indices to sort your images:
  sorted_imgs = np.array([imgs[idx_i] for idx_i in idxs])

  return sorted_imgs


def prepare_all(dir_name,N):

  classes = [d for d in os.listdir('./'+dir_name+'/') if os.path.isdir(os.path.join('./'+dir_name+'/', d))]
  temp=0
  temp_=0
  dirs = [d for d in os.listdir('./'+dir_name+'/') if os.path.isdir(os.path.join('./'+dir_name+'/', d))]
  
  print(dirs)
  print(len(classes))
  print(classes[1])
  tensors =[]
  Y=np.zeros((N*len(classes),len(classes)))
  for i in range(len(classes)):
    '''
    if(i<len(classes)-1):
      X=prepare(classes[i],N)
      X1=prepare(classes[i+1],N)
      X=np.concatenate(prepare(X,X1),axis=0)
    '''
    tensors.append(prepare(classes[i],N))
    temp=temp+N
    Y[temp_:temp,i]=1
    temp_=temp_+N

  X = np.concatenate((tensors[0],tensors[1]),axis=0)

  for i in range(len(classes)-2):
    X=np.concatenate((X,tensors[2+i]),axis=0)
  
  return X,Y






