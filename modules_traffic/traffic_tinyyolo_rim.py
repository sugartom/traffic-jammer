# Darkflow should be installed from: https://github.com/sugartom/darkflow
from darkflow.net.build import TFNet

import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util
import cv2
import tensorflow as tf

# Place your downloaded cfg and weights under "checkpoints/"
YOLO_CONFIG = '/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/cfg'
YOLO_MODEL = '/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/cfg/tiny-yolo-voc.cfg'
YOLO_WEIGHTS = '/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/bin/tiny-yolo-voc.weights'
YOLO_THRES = 0.4

class TrafficTinyYolo:

  @staticmethod
  def Setup():
    opt = { 
            "config": YOLO_CONFIG,  
            "model": YOLO_MODEL, 
            "load": YOLO_WEIGHTS, 
            "threshold": YOLO_THRES
          }
    TrafficTinyYolo.tfnet = TFNet(opt)

  def PreProcess(self, request, istub, grpc_flag):
    if (grpc_flag):
      self.image = tensor_util.MakeNdarray(request.inputs["client_input"])
    else:
      self.image = request["client_input"]

    self.istub = istub

  def Apply(self):
    dets = TrafficTinyYolo.tfnet.return_predict(self.image, "traffic_tinyyolo", self.istub)

    output = ""
    for d in dets:
      output += "%s|%s|%s|%s|%s|%s-" % (str(d['topleft']['x']), str(d['topleft']['y']), str(d['bottomright']['x']), str(d['bottomright']['y']), str(d['confidence']), str(d['label']))
    self.output = output[:-1]

  def PostProcess(self, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      next_request.inputs["client_input"].CopyFrom(
        tf.make_tensor_proto(self.image))
      next_request.inputs["output"].CopyFrom(
        tf.make_tensor_proto(self.output))
    else:
      next_request = dict()
      next_request["client_input"] = self.image
      next_request["output"] = self.output
    return next_request

# import grpc
# from tensorflow_serving.apis import prediction_service_pb2_grpc

# ichannel = grpc.insecure_channel('0.0.0.0:8500')
# istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

# traffic_yolo = TrafficTinyYolo()
# traffic_yolo.Setup()

# image = cv2.imread("/home/yitao/Downloads/person.jpg")
# print(image.shape)
# request = dict()
# request["client_input"] = image

# traffic_yolo.PreProcess(request, istub, False)
# traffic_yolo.Apply()
# next_request = traffic_yolo.PostProcess(False)

# print(next_request["output"])

