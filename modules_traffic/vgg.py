import cv2

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2

import skimage
import skimage.io
import skimage.transform
import numpy as np

class Vgg:

  @staticmethod
  def Setup():
    pass

  def load_image(self, img):
    # load image
    # img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img

  def PreProcess(self, request_input, istub, grpc_flag):
    if (grpc_flag):
      self.request_input = str(tensor_util.MakeNdarray(request_input.inputs["client_input"]))
      self.image = cv2.imdecode(np.fromstring(self.request_input, dtype = np.uint8), -1)
    else:
      self.image = request_input['client_input']

    self.istub = istub

    self.image = self.load_image(self.image)
    # print(self.image.shape)
    # self.image = [cv2.imencode('.jpg', self.image)[1].tostring()]
    # self.image = [self.image.reshape(1, 224, 224, 3)]

  def Apply(self):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'vgg_model'
    request.model_spec.signature_name = 'predict_images'
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(np.float32(self.image), shape=[1, 224, 224, 3]))
    
    result = self.istub.Predict(request, 10.0)
    self.output = "FaceDone"

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