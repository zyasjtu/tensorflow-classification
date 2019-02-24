import os
import sys

import utils

cfg = utils.get_cfg(sys.argv[1])
base_dir = cfg['databasedir']
class_names = os.listdir(base_dir)

valid = open(cfg['valpath'], 'w')
train = open(cfg['trainpath'], 'w')

for i in range(len(class_names)):
    sub_dir = base_dir + '/' + class_names[i] + '/'
    file_names = os.listdir(sub_dir)
    for j in range(len(os.listdir(sub_dir))):
        if ((j + 1) % 3 == 0):
            valid.writelines(sub_dir + file_names[j] + ',' + str(i) + '\n')
        else:
            train.writelines(sub_dir + file_names[j] + ',' + str(i) + '\n')
valid.close()
train.close()
