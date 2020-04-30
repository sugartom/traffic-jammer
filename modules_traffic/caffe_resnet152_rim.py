import cv2

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2

class CaffeResnet152:

  @staticmethod
  def Setup():
    pass

  def PreProcess(self, request, istub, grpc_flag):
    if (grpc_flag):
      self.image = tensor_util.MakeNdarray(request.inputs["client_input"])
    else:
      self.image = request["client_input"]

    self.istub = istub

    self.image = [cv2.imencode('.jpg', self.image)[1].tostring()]

  def Apply(self):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'traffic_resnet152'
    request.model_spec.signature_name = 'predict_images'
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(self.image, shape=[len(self.image)]))
    
    result = self.istub.Predict(request, 10.0)
    self.output = "FaceDone"

  def PostProcess(self, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      next_request.inputs['FINAL'].CopyFrom(
          tf.make_tensor_proto(self.output))
    else:
      next_request = dict()
      next_request["FINAL"] = self.output
    return next_request
