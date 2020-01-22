
import cv2
import tensorflow as tf
import numpy as np
import glob
import os
import time
import argparse
import configparser

from auto_pose.ae import factory, utils



parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
parser.add_argument("-f", "--file_str", required=True, help='folder or filename to image(s)')
# parser.add_argument("-gt_bb", action='store_true', default=False)
arguments = parser.parse_args()
full_name = arguments.experiment_name.split('/')
experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''

print('experiment name:  ', experiment_name)
print('experiment group:  ', experiment_group)


file_str = arguments.file_str
if os.path.isdir(file_str):
    files = sorted(glob.glob(os.path.join(str(file_str),'*.png'))+glob.glob(os.path.join(str(file_str),'*.jpg'))+glob.glob(os.path.join(str(file_str),'*.JPG')))
else:
    files = [file_str]

workspace_path = os.environ.get('AE_WORKSPACE_PATH')
if workspace_path == None:
    print('Please define a workspace path:\n')
    print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
    exit(-1)
log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
ckpt_dir = utils.get_checkpoint_dir(log_dir)

start_time = time.time()
encoder = factory.build_codebook_from_name(experiment_name, experiment_group, return_encoder=True)
end_time = time.time()
print("encoder loading: ", str(end_time - start_time))


with tf.Session() as sess:

    start_time = time.time()
    factory.restore_checkpoint(sess, tf.train.Saver(), ckpt_dir)
    end_time = time.time()
    print("restoring checkpoint: ", str(end_time - start_time))

    # for i in range(1, 8):
    for file in files:

        im = cv2.imread(file)
        im = cv2.resize(im, (256, 256))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = np.expand_dims(im, axis=2)

        start_time = time.time()
        latent_vector = encoder.latent_vector(sess, im)
        end_time = time.time()
        print('latent vector: ', latent_vector)
        print("inference time: ", int(1000 * (end_time - start_time)) / 1000., "    fps: ",
              int(1 / (end_time - start_time)))
