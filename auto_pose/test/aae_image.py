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

codebook, dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)


with tf.Session() as sess:

    factory.restore_checkpoint(sess, tf.train.Saver(), ckpt_dir)

    video_path = '/test_7.MOV'
    # codec = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(os.path.join(workspace_path, "out_" + os.path.basename(os.path.splitext(video_path)[0]) + ".avi"), codec, 30, (512, 256))
    cap = cv2.VideoCapture(video_path)
    successive_falses = 0
    while cap.isOpened() and successive_falses < 30:
        ret, im = cap.read()
        if ret:
            successive_falses = max(0, successive_falses - 1)
            H, W = im.shape[:2]
            mmin, mmax = min(H, W), max(H, W)
            diff = mmax - mmin
            diff = diff // 3
            offset = 0
            im = im[:, offset: offset + mmin + diff]

            im = cv2.resize(im, (256,256))
            start_time = time.time()
            R = codebook.nearest_rotation(sess, im)
            end_time = time.time()
            print("inference time: ", end_time - start_time)
            print(R)
            # pred_view = dataset.render_rot( R,downSample = 1)

            # out_img = np.hstack([pred_view, im])
            # out.write(out_img)

            # cv2.imshow('resized img', cv2.resize(im/255.,(256,256)))
            # cv2.imshow('pred_view', cv2.resize(pred_view,(256,256)))
            # cv2.imshow('both together', out_img)
            # cv2.waitKey(20)
        else:
            successive_falses = successive_falses + 1

    cap.release()
    # out.release()

    cv2.destroyAllWindows()

