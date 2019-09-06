import time
import sys
import cv2

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/')
from modules_traffic.traffic_yolo_tf import YOLO
from modules_traffic.inception_tf import Inception
from modules_traffic.resnet_tf import Resnet

yolo = YOLO()
yolo.Setup()
inception = Inception()
inception.Setup()
resnet = Resnet()
resnet.Setup()

frame_num = 10
while frame_num:
  frame_num -= 1

  start = time.time()

  image = cv2.imread("image1.jpg")

  # here yolo's output is still the original input image
  yolo.PreProcess(image)
  yolo.Apply()
  yolo_output = yolo.PostProcess()

  # feed yolo's output to inception
  inception.PreProcess(yolo_output)
  inception.Apply()
  inception_output = inception.PostProcess()

  # feed yolo's output to resnet
  resnet.PreProcess(yolo_output)
  resnet.Apply()
  resnet_output = resnet.PostProcess()

  end = time.time()
  print('duration = %s' % (end - start))
