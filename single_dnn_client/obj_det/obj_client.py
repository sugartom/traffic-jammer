# https://towardsdatascience.com/productising-tensorflow-keras-models-via-tensorflow-serving-69e191cb1f37
# python -m grpc.tools.protoc --python_out=. --grpc_python_out=. -I. string_int_label_map.proto

# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import sys
import time
import cv2
import numpy as np
import grpc
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import string_int_label_map_pb2 as labelmap
from google.protobuf import text_format

from image_preprocessor import decode_image_opencv

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/single_dnn_client/dog.jpg', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

def box_normal_to_pixel(box, dim,scalefactor=1):
  # https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py
  height, width = dim[0], dim[1]
  ymin = int(box[0]*height*scalefactor)
  xmin = int(box[1]*width*scalefactor)

  ymax = int(box[2]*height*scalefactor)
  xmax= int(box[3]*width*scalefactor)
  return np.array([xmin,ymin,xmax,ymax])   

def get_label(index):
  global _label_map
  return _label_map.item[index].display_name

def main(_):
  if (sys.argv[1] == "mobilenet"):
    model_name = "ssd_mobilenet_v1_coco"
  elif (sys.argv[1] == "inception"):
    model_name = "ssd_inception_v2_coco"
  else:
    model_name = "ssd_resnet50_v1_fpn"

  s = open('/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/single_dnn_client/obj_det/mscoco_complete_label_map.pbtxt','r').read()
  mymap =labelmap.StringIntLabelMap()
  global _label_map 
  _label_map = text_format.Parse(s,mymap)

  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  run_num = 10
  batch_size = 1
  for i in range(run_num):
    start = time.time()

    image, org = decode_image_opencv(FLAGS.image)
    _draw = org.copy()

    image = image.astype(np.uint8)
    inputs = image
    for i in range(batch_size - 1):
      inputs = np.append(inputs, image, axis = 0)

    request = predict_pb2.PredictRequest()    
    request.model_spec.name = model_name
    request.model_spec.signature_name = 'serving_default'
    request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto
           (inputs, shape=inputs.shape))

    result = stub.Predict(request, 10.0)
    # print(result)
    boxes = result.outputs['detection_boxes']
    scores = result.outputs['detection_scores']
    labels = result.outputs['detection_classes']
    num_detections= result.outputs['num_detections']

    # print("???")
    # print(boxes)

    boxes= tf.make_ndarray(boxes)
    scores= tf.make_ndarray(scores)
    labels= tf.make_ndarray(labels)
    num_detections= tf.make_ndarray(num_detections)

    # print("boxes output",(boxes).shape)
    # print("scores output",(scores).shape)
    # print("labels output",(labels).shape)
    # print('num_detections',num_detections[0])

    # # visualize detections hints from 
    # # # https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

    # for box, score, label in zip(boxes[0], scores[0], labels[0]):
    #   # scores are sorted so we can break
    #   if score < 0.3:
    #       break
    #   #dim = image.shape[0:2]
    #   dim = _draw.shape
    #   #print("Label-raw",labels_to_names[label-1]," at ",box," Score ",score)
    #   box = box_normal_to_pixel(box, dim)
    #   b = box.astype(int)
    #   class_label = get_label(int(label))
    #   print("Label",class_label ," at ",b," Score ",score)
    #   # draw the image and write out
    #   cv2.rectangle(_draw,(b[0],b[1]),(b[2],b[3]),(0,0,255),1)
    #   cv2.putText(_draw,class_label + "-"+str(round(score,2)), (b[0]+2,b[1]+8),\
    #      cv2.FONT_HERSHEY_SIMPLEX, .45, (0,0,255))

    # cv2.imshow("test", _draw)
    # cv2.waitKey(0)

    end = time.time()
    duration = end - start
    print("duration = %s sec" % str(duration))

if __name__ == '__main__':
  tf.app.run()