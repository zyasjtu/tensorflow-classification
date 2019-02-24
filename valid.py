import os
import sys

import numpy as np
import tensorflow as tf

import networks.vgg as vgg
import utils

cfg = utils.get_cfg(sys.argv[1])
classes_name = os.listdir(cfg['databasedir'])


def test_vgg(fn, cfg):
    print('test_vgg', fn)
    x = tf.placeholder(dtype='float32', shape=[None, cfg['height'], cfg['width'], 3])
    vgg16 = vgg.Vgg(x, len(classes_name), "true" == cfg['isvgg19'], cfg['modelpath'])
    prob = vgg16.build(False)
    img = utils.load_image(fn, cfg['width'], cfg['height'])
    img = img - np.array(cfg['mean'])
    batch1 = img.reshape([1, cfg['height'], cfg['width'], 3])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # vgg16.loadModel(sess)
        tf.train.Saver().restore(sess, cfg['finetunedpath'])
        out = sess.run(prob, feed_dict={x: batch1})[0][0]
        classes_num = len(out)
        print(classes_num)
        pred = np.argsort(out)
        for i in range(5):
            index = classes_num - i - 1
            print(pred[index], out[pred[index]], classes_name[pred[index]])


if __name__ == '__main__':
    test_vgg(sys.argv[2], cfg)
