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

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import skimage
import skimage.io
import skimage.transform
import numpy as np

import time

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def load_image(path):
    # load image
    img = skimage.io.imread(path)
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

def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1

# def get_batch(batch_size):
#   img = load_image("/home/yitao/Documents/fun-project/tensorflow-vgg/test_data/tiger.jpeg").reshape((1, 224, 224, 3))
#   batch = np.concatenate((img, img, img), 0)

#   return batch

# def get_batch(batch_size):
#   images = []
#   for i in range(batch_size):
#     img = load_image("/home/yitao/Documents/fun-project/tensorflow-vgg/test_data/tiger.jpeg")
#     images.append(img.reshape((1, 224, 224, 3)))
  
#   batch = np.concatenate(images, 0)
#   return batch

def myFuncWarmUp(stub, i):
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'vgg_model'
  request.model_spec.signature_name = 'predict_images'

  batchSize = 1
  durationSum = 0.0
  runNum = 13

  for k in range(runNum):
    image_data = []
    start = time.time()
    for j in range(batchSize):
      # image_name = "/home/yitao/Downloads/inception-input/%s/dog-%s.jpg" % (str(i % 100).zfill(3), str(j).zfill(3))
      image_name = "/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/single_dnn_client/dog.jpg"
      img = load_image(image_name)
      image_data.append(img.reshape((1, 224, 224, 3)))
    batch = np.concatenate(image_data, 0)
    # print(batch.shape)
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(np.float32(image_data[0]), shape=[batchSize, 224, 224, 3]))

    tmp_result = stub.Predict(request, 10.0)  # 5 seconds
    # print(len(tmp_result.outputs["scores"].float_val))
    end = time.time()
    duration = (end - start)
    print("it takes %s sec" % str(duration))
    if (k != 0 and k != 3 and k != 8):
      durationSum += duration

  print("[Warm up] on average, it takes %s sec to run a batch of %d images over %d runs" % (str(durationSum / (runNum - 3)), batchSize, (runNum - 3)))

def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  myFuncWarmUp(stub, 0)


if __name__ == '__main__':
  tf.app.run()