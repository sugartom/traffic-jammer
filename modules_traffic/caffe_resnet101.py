import cv2

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2

class CaffeResnet101:

  @staticmethod
  def Setup():
    pass

  def PreProcess(self, request_input, istub, grpc_flag):
    if (grpc_flag):
      self.request_input = str(tensor_util.MakeNdarray(request_input.inputs["client_input"]))
      self.image = cv2.imdecode(np.fromstring(self.request_input, dtype = np.uint8), -1)
    else:
      self.image = request_input['client_input']

    self.istub = istub

    self.image = [cv2.imencode('.jpg', self.image)[1].tostring()]

  def Apply(self):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'caffe_resnet101'
    request.model_spec.signature_name = 'predict_images'
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(self.image, shape=[len(self.image)]))
    
    result = self.istub.Predict(request, 10.0)
    self.output = "CarDone"

  def PostProcess(self, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      next_request.inputs['FINAL'].CopyFrom(
          tf.make_tensor_proto(self.output))
      return next_request
    else:
      result = dict()
      result["FINAL"] = self.output
      return result