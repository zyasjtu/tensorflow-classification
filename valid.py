import os
import sys

import numpy as np
import tensorflow as tf

import networks.vgg as vgg
import utils

classes_name = os.listdir('/volume/tensorflow-classification/data/logos')
valid_set = open('/volume/tensorflow-classification/data/valid.txt', 'r')
classes_id = [str(x) for x in range(len(classes_name))]
mean = np.array([103.939, 116.779, 123.68])


def valid(isvgg19):
	print('prepare network...')
	x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
	if isvgg19 == 'true':
		vgg_network = vgg.Vgg(x, len(classes_id), True, './models/vgg19.npy')
	else:
		vgg_network = vgg.Vgg(x, len(classes_id), False, './models/vgg16.npy')
	predictions, logits, feature = vgg_network.build(False)
	feat_mat = np.loadtxt('feat_mat.txt')

	lines = valid_set.readlines()

	print('start...')
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		vgg_network.loadModel(sess, True)

		correct_count = 0
		current_count = 0
		for line in lines:
			line = line.strip('\n')
			split_idx = line.find(',')
			img_path = '/volume/tensorflow-classification/' + line[0:split_idx]
			img_label = line[split_idx + 1:]
			img = utils.load_image(img_path, 224, 224)
			img = img - mean
			batch1 = img.reshape([1, 224, 224, 3])
			feat_vec = sess.run(feature, feed_dict={x: batch1})[0]

			feat_dif_norm = []
			for i in range(len(feat_mat)):
				feat_dif = feat_vec - feat_mat[i]
				feat_dif_norm.append(np.linalg.norm(feat_dif))

			norm_min = min(feat_dif_norm)
			min_index = feat_dif_norm.index(norm_min)
			current_count += 1
			if (img_label == str(min_index)):
				correct_count += 1
			else:
				print(img_path, img_label, classes_name[int(img_label)], norm_min, min_index, classes_name[min_index],
					  correct_count, current_count)
		print(correct_count, current_count, float(correct_count) / len(lines))
				

if __name__ == '__main__':
	valid(sys.argv[1])
