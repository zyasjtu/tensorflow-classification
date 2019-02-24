import os
import sys

import numpy as np
import tensorflow as tf

import networks.vgg as vgg
import utils

classes_name = os.listdir('/volume/tensorflow-classification/data/logos')
train_set = open('/volume/tensorflow-classification/data/train.txt', 'r')
classes_id = [str(x) for x in range(len(classes_name))]
mean = np.array([103.939, 116.779, 123.68])


def train(isvgg19):
	print('prepare network...')
	x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
	if isvgg19 == 'true':
		vgg_network = vgg.Vgg(x, len(classes_id), True, './models/vgg19.npy')
	else:
		vgg_network = vgg.Vgg(x, len(classes_id), False, './models/vgg16.npy')
	predictions, logits, feature = vgg_network.build(False)

	feat_mat = []
	feat_mean = [0 for i in range(4096)]
	lines = train_set.readlines()

	print('start...')
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		vgg_network.loadModel(sess, True)

		cur_label = '0'
		cur_count = 0
		for line in lines:
			line = line.strip('\n')
			split_idx = line.find(',')
			img_path = '/volume/tensorflow-classification/' + line[0:split_idx]
			img_label = line[split_idx + 1:]
			img = utils.load_image(img_path, 224, 224)
			img = img - mean
			batch1 = img.reshape([1, 224, 224, 3])
			feat_vec = sess.run(feature, feed_dict={x: batch1})[0]
			if (cur_label == img_label):
				cur_count += 1
				feat_mean += feat_vec
			else:
				feat_mean = feat_mean / cur_count
				feat_mat.append(feat_mean)
				cur_count = 1
				cur_label = img_label
				feat_mean = feat_vec
			print(line, cur_label, cur_count)
			print(len(feat_mat), '0' if len(feat_mat) == 0 else len(feat_mat[len(feat_mat) - 1]))
			assert (cur_label == str(len(feat_mat)))
		feat_mean = feat_mean / cur_count
		feat_mat.append(feat_mean)
		print(len(feat_mat), len(feat_mat[len(feat_mat) - 1]))
		np.savetxt('feat_mat.txt', feat_mat)

if __name__ == '__main__':
	train(sys.argv[1])
