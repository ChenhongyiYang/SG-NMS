import sys
sys.path.append('/home/grad3/hongyi/rfcn_occ/rfcn_occ')
import numpy as np
import argparse
import os

import tensorflow as tf
from nets.ROIPooling.ROIPoolingWrapper import positionSensitiveRoiPooling_unnorm , positionSensitiveRoiPooling_norm


parser = argparse.ArgumentParser(description='Position Sensitive RoI Pooling')
parser.add_argument('--gpu', required=True, type=str, help='gpu inds')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print('==> GPU index:',args.gpu)




def test_his():
	with tf.Session() as sess:
		img = np.zeros((1,8,8,9), np.float32)
		boxes = tf.constant([[0,0,2*16,5*16]], dtype=tf.float32)
		print(boxes.get_shape().as_list())

		yOffset=0
		xOffset=0
		chOffset=0
		img[0,yOffset+0:yOffset+1,xOffset+0:xOffset+1,chOffset+0:chOffset+1]=1;
		#img[:,:,:,:]=1
		p = tf.placeholder(tf.float32, shape=img.shape)

		np.set_printoptions(threshold=5000, linewidth=150)

		pooled=positionSensitiveRoiPooling_unnorm(p, boxes)
		print(sess.run(pooled, feed_dict={p: img}))

		loss = tf.reduce_sum(pooled)

		g = tf.gradients(loss, p)

		print(img)
		print(sess.run(g, feed_dict={p: img})[0])
		print(sess.run(g, feed_dict={p: img})[0][:,:,:,1])



def test_mine():
	with tf.Session() as sess:
		img = np.arange(81).reshape(1,9,9,1)
		img = np.repeat(img, 12, axis=3)

		#boxes = tf.constant([[2./9-0.001, 2./9-0.001, 7./9-0.001, 4./9-0.001],[3./9, 3./9, 6./9, 7./9]], dtype=tf.float32)
		#[x, y, x, y]
		boxes = tf.constant([[2. / 9 - 0.001, 1. / 9 - 0.001, 4. / 9 - 0.001, 4. / 9 - 0.001]], dtype=tf.float32)
		boxes = tf.constant([[2. / 9 - 0.001, 1. / 9 - 0.001, 2. / 9 - 0.002, 1. / 9 - 0.002]], dtype=tf.float32)
		p = tf.placeholder(tf.float32, shape=img.shape)
		pooled= positionSensitiveRoiPooling_norm(p, boxes, num_channel=1, feature_shape=(9,9), K=(4,3))
		_pooled = sess.run([pooled], feed_dict={p: img})[0]
		print(_pooled.shape)
		print(_pooled)




if __name__ == "__main__":
	test_mine()


