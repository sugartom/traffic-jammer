# Darkflow should be installed from: https://github.com/sugartom/darkflow
from darkflow.net.build import TFNet

from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util
import tensorflow as tf

# Place your downloaded cfg and weights under "checkpoints/"
YOLO_CONFIG = '/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/cfg'
YOLO_MODEL = '/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/cfg/yolo.cfg'
YOLO_WEIGHTS = '/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/bin/yolo.weights'
YOLO_THRES = 0.4
YOLO_PEOPLE_LABEL = 'person'

class TrafficYolo:

  @staticmethod
  def Setup():
    opt = { 
            "config": YOLO_CONFIG,  
            "model": YOLO_MODEL, 
            "load": YOLO_WEIGHTS, 
            "threshold": YOLO_THRES
          }
    TrafficYolo.tfnet = TFNet(opt)

  def PreProcess(self, request, grpc_flag):
    request_dict = dict()

    if (grpc_flag):
      request_dict["client_input"] = tensor_util.MakeNdarray(request.inputs["client_input"])
    else:
      request_dict["client_input"] = request["client_input"]

    return request_dict  

  def Apply(self, request_dict, istub):
    dets = TrafficYolo.tfnet.return_predict(request_dict["client_input"], "traffic_yolo", istub)

    output = ""
    for d in dets:
      if d['label'] != YOLO_PEOPLE_LABEL:
        continue
      output += "%s|%s|%s|%s|%s|%s-" % (str(d['topleft']['x']), str(d['topleft']['y']), str(d['bottomright']['x']), str(d['bottomright']['y']), str(d['confidence']), str(d['label']))
    objdet_output = output[:-1]

    next_request_dict = dict()
    next_request_dict["client_input"] = request_dict["client_input"]
    next_request_dict["objdet_output"] = objdet_output

    return [next_request_dict]

  def PostProcess(self, next_request_dict, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      next_request.inputs["client_input"].CopyFrom(
        tf.make_tensor_proto(next_request_dict["client_input"]))
      next_request.inputs["objdet_output"].CopyFrom(
        tf.make_tensor_proto(next_request_dict["objdet_output"]))
    else:
      next_request = dict()
      next_request["client_input"] = next_request_dict["client_input"]
      next_request["objdet_output"] = next_request_dict["objdet_output"]
    return next_request
