import cv2
import numpy as np
import os

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2

import string_int_label_map_pb2 as labelmap
from google.protobuf import text_format

class TrafficInception:

  @staticmethod
  def Setup():
    s = open('%s/modules_traffic/mscoco_complete_label_map.pbtxt' % os.environ['TRAFFIC_JAMMER_PATH'], 'r').read()
    mymap = labelmap.StringIntLabelMap()
    TrafficInception._label_map = text_format.Parse(s, mymap)

  def decode_image_opencv(self, image, max_height = 800, swapRB = True, imagenet_mean = (0, 0, 0)):
    # image = cv2.imread(img_path, 1)
    (h, w) = image.shape[:2]
    image = self.image_resize(image, height = max_height)
    org  = image
    image = cv2.dnn.blobFromImage(image, scalefactor=1.0, mean = imagenet_mean, swapRB = swapRB)
    image = np.transpose(image, (0, 2, 3, 1))
    return image, org

  def image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

  def box_normal_to_pixel(self, box, dim, scalefactor = 1):
    # https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py
    height, width = dim[0], dim[1]
    ymin = int(box[0] * height * scalefactor)
    xmin = int(box[1] * width * scalefactor)

    ymax = int(box[2] * height * scalefactor)
    xmax= int(box[3] * width * scalefactor)
    return np.array([xmin, ymin, xmax, ymax])

  def get_label(self, index):
    return TrafficInception._label_map.item[index].display_name

  def PreProcess(self, request, istub, grpc_flag):
    if (grpc_flag):
      self.image = tensor_util.MakeNdarray(request.inputs["client_input"])
    else:
      self.image = request["client_input"]

    self.istub = istub

    self.input, self.org = self.decode_image_opencv(self.image)
    self.input = self.input.astype(np.uint8)

  def Apply(self):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'traffic_inception'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['inputs'].CopyFrom(
      tf.contrib.util.make_tensor_proto(self.input, shape=self.input.shape))
    
    result = self.istub.Predict(request, 10.0)

    boxes = tf.make_ndarray(result.outputs['detection_boxes'])
    scores = tf.make_ndarray(result.outputs['detection_scores'])
    labels = tf.make_ndarray(result.outputs['detection_classes'])
    # num_detections= tf.make_ndarray(result.outputs['num_detections'])

    # _draw = self.org.copy()
    # print(_draw.shape)
    # for box, score, label in zip(boxes[0], scores[0], labels[0]):
    #   # scores are sorted so we can break
    #   if score < 0.3:
    #       break
    #   dim = _draw.shape
    #   box = self.box_normal_to_pixel(box, dim)
    #   b = box.astype(int)
    #   class_label = self.get_label(int(label))
    #   print("Label = %s at %s with score of %s" % (class_label, b, score))
    #   # draw the image and write out
    #   cv2.rectangle(_draw,(b[0],b[1]),(b[2],b[3]),(0,0,255),1)
    #   cv2.putText(_draw,class_label + "-"+str(round(score,2)), (b[0]+2,b[1]+8),\
    #      cv2.FONT_HERSHEY_SIMPLEX, .45, (0,0,255))

    # cv2.imshow("test", _draw)
    # cv2.waitKey(0)

  def PostProcess(self, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      next_request.inputs["client_input"].CopyFrom(
        tf.make_tensor_proto(self.image))
      next_request.inputs["FINAL"].CopyFrom(
        tf.make_tensor_proto("Done"))
    else:
      next_request = dict()
      next_request["client_input"] = self.image
      next_request["FINAL"] = "Done"
    return next_request

unit_test = False
if (unit_test):
  import grpc
  import time
  from tensorflow_serving.apis import prediction_service_pb2_grpc

  ichannel = grpc.insecure_channel('0.0.0.0:8500')
  istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

  traffic_inception = TrafficInception()
  traffic_inception.Setup()

  image = cv2.imread("/home/yitao/Downloads/person.jpg")
  print(image.shape)
  request = dict()
  request["client_input"] = image

  for i in range(10):
    start = time.time()
    traffic_inception.PreProcess(request, istub, False)
    traffic_inception.Apply()
    next_request = traffic_inception.PostProcess(False)
    end = time.time()
    print("duration = %s" % (end - start))
