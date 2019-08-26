import cv2
import numpy as np

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2

class Mobilenet:

  @staticmethod
  def Setup():
    pass

  def decode_image_opencv(self, image, max_height = 800, swapRB = True, imagenet_mean = (0, 0, 0)):
    # image = cv2.imread(img_path, 1)
    (h, w) = image.shape[:2]
    image = self.image_resize(image, height = max_height)
    org  = image
    image = cv2.dnn.blobFromImage(image, scalefactor=1.0, mean = imagenet_mean, swapRB = swapRB)
    image = np.transpose(image, (0, 2, 3, 1))
    return image,org

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

  def PreProcess(self, request_input, istub, grpc_flag):
    if (grpc_flag):
      self.request_input = str(tensor_util.MakeNdarray(request_input.inputs["client_input"]))
      self.image = cv2.imdecode(np.fromstring(self.request_input, dtype = np.uint8), -1)
    else:
      self.image = request_input['client_input']

    self.istub = istub

    # self.image = [cv2.imencode('.jpg', self.image)[1].tostring()]
    self.image, _ = self.decode_image_opencv(self.image)
    self.image = self.image.astype(np.uint8)

  def Apply(self):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'traffic_mobilenet'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(self.image, shape=self.image.shape))
    
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