"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.08
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: Testing script
"""
#！/usr/bin/env python3
import argparse
import sys
import os
import time
import rospy
from sensor_msgs import PointCloud2
from sensor_msgs import point_cloud2

from easydict import EasyDict as edict
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import torch
import numpy as np

sys.path.append('../')

import config.kitti_config as cnf
from data_process import kitti_data_utils, kitti_bev_utils
from data_process.my_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder
from utils.evaluation_utils import post_processing, rescale_boxes, post_processing_v2
from utils.misc import time_synchronized
from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format


def parse_test_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for Complex YOLO Implementation')
    parser.add_argument('--saved_fn', type=str, default='complexer_yolov4', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/complex_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf_thresh', type=float, default=0.5,
                        help='the threshold for conf')
    parser.add_argument('--nms_thresh', type=float, default=0.1,
                        help='the threshold for conf')

    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demostration')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_complexer_yolov4', metavar='PATH',
                        help='the video filename if the output format is video')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.working_dir = '../'
    configs.dataset_dir = '/media/wx/File/labeldata/mydata'

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.working_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs


class Detection:
    def __init__(self, n, configs):
        self.node = n
        #self.pub = rospy.Publisher('tracking_result', Object_with_id, queue_size=10)
        self.model = create_model(configs)
        self.model.print_network()
        print('\n\n' + '-*=' * 30 + '\n\n')
        assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
        self.model.load_state_dict(torch.load(configs.pretrained_path,map_location='cuda:0'))
        configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
        self.model = self.model.to(device=configs.device)
        self.model.eval()

        rospy.Subscriber("detect_object", PointCloud2, self.callback) # your cloud topic name
        rospy.spin()
    def callback(self,data):
        rospy.loginfo("detection")
        with torch.no_grad():
            gen = point_cloud2.read_points(data)
            for idx, p in enumerate(gen):
                print(idx)
                print(p)

            b = kitti_bev_utils.removePoints(gen, cnf.boundary)
            imgs_bev = kitti_bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)

            input_imgs = imgs_bev.to(device=configs.device).float()
            t1 = time_synchronized()
            outputs = self.model(input_imgs)
            t2 = time_synchronized()
            detections = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)

            img_detections = []  # Stores detections for each image index
            img_detections.extend(detections)

            img_bev = imgs_bev.squeeze() * 255
            img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
            img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))
            for detections in img_detections:
                if detections is None:
                    continue
                # Rescale boxes to original image
                detections = rescale_boxes(detections, configs.img_size, img_bev.shape[:2])
                for x, y, w, l, im, re, *_, cls_pred in detections:
                    yaw = np.arctan2(im, re)
                    # Draw rotated box
                    kitti_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])

            img_bev = cv2.flip(cv2.flip(img_bev, 0), 1)
            out_img = img_bev
            cv2.imshow('test-img', out_img)
            cv2.waitKey(1)

if __name__ == '__main__':
    configs = parse_test_configs()
    configs.distributed = False  # For testing
    node = rospy.init_node("detection_listener",anonymous=True) #rosnode
    Detection(node, configs)
    rospy.spin()
