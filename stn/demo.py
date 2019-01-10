from scipy import ndimage
import tensorflow as tf
from STN import transformer
import numpy as np
import matplotlib.pyplot as plt
import cv2

im = cv2.imread('./0.jpg')  # 改为你自己要测试的图片路径
im = im / 255.
# im=tf.reshape(im, [1,1200,1600,3])

im = np.expand_dims(im, axis=0)

im = im.astype('float32')
print('img-over')
out_size = (600, 800)
batch = np.append(im, im, axis=0)
batch = np.append(batch, im, axis=0)
num_batch = 3

x = tf.placeholder(tf.float32, [None, 1200, 1600, 3])
x = tf.cast(batch, 'float32')
print('begin---')
with tf.variable_scope('spatial_transformer_0'):
	n_fc = 6
	w_fc1 = tf.Variable(tf.Variable(tf.zeros([1200 * 1600 * 3, n_fc]), name='W_fc1'))
	initial = np.array([[0.5, 0, 0], [0, 0.5, 0]])
	initial = initial.astype('float32')
	initial = initial.flatten()
	
	b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
	
	h_fc1 = tf.matmul(tf.zeros([num_batch, 1200 * 1600 * 3]), w_fc1) + b_fc1
	
	print(x, h_fc1, out_size)
	
	h_trans = transformer(x, h_fc1, out_size)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
y = sess.run(h_trans, feed_dict={x: batch})
plt.imshow(y[0])
plt.show()