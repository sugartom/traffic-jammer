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

"""A client that talks to tensorflow_model_server loaded with mnist model.

The client downloads test images of mnist data set, queries the service with
such test images to get predictions, and calculates the inference error rate.

Typical usage example:

    mnist_client.py --num_tests=100 --server=localhost:9000
"""

from __future__ import print_function

import time
import sys
import threading

import numpy as np

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import mnist_input_data


tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS


# class _ResultCounter(object):
#   """Counter for the prediction results."""

#   def __init__(self, num_tests, concurrency):
#     self._num_tests = num_tests
#     self._concurrency = concurrency
#     self._error = 0
#     self._done = 0
#     self._active = 0
#     self._condition = threading.Condition()

#   def inc_error(self):
#     with self._condition:
#       self._error += 1

#   def inc_done(self):
#     with self._condition:
#       self._done += 1
#       self._condition.notify()

#   def dec_active(self):
#     with self._condition:
#       self._active -= 1
#       self._condition.notify()

#   def get_error_rate(self):
#     with self._condition:
#       while self._done != self._num_tests:
#         self._condition.wait()
#       return self._error / float(self._num_tests)

#   def throttle(self):
#     with self._condition:
#       while self._active == self._concurrency:
#         self._condition.wait()
#       self._active += 1


# def _create_rpc_callback(label, result_counter):
#   """Creates RPC callback function.

#   Args:
#     label: The correct label for the predicted example.
#     result_counter: Counter for the prediction result.
#   Returns:
#     The callback function.
#   """
#   def _callback(result_future):
#     """Callback function.

#     Calculates the statistics for the prediction result.

#     Args:
#       result_future: Result future of the RPC.
#     """
#     exception = result_future.exception()
#     if exception:
#       result_counter.inc_error()
#       print(exception)
#     else:
#       sys.stdout.write('.')
#       sys.stdout.flush()
#       response = numpy.array(
#           result_future.result().outputs['scores'].float_val)
#       prediction = numpy.argmax(response)
#       if label != prediction:
#         result_counter.inc_error()
#     result_counter.inc_done()
#     result_counter.dec_active()
#   return _callback


# def do_inference(hostport, work_dir, concurrency, num_tests):
#   """Tests PredictionService with concurrent requests.

#   Args:
#     hostport: Host:port address of the PredictionService.
#     work_dir: The full path of working directory for test data set.
#     concurrency: Maximum number of concurrent requests.
#     num_tests: Number of test images to use.

#   Returns:
#     The classification error rate.

#   Raises:
#     IOError: An error occurred processing test data set.
#   """
#   test_data_set = mnist_input_data.read_data_sets(work_dir).test
#   host, port = hostport.split(':')
#   channel = implementations.insecure_channel(host, int(port))
#   stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

#   start = time.time()
#   for _ in range(num_tests):
#     request = predict_pb2.PredictRequest()
#     request.model_spec.name = 'caffe_mnist'
#     request.model_spec.signature_name = 'predict_images'
#     image, label = test_data_set.next_batch(5)
#     # tmp_image, _ = test_data_set.next_batch(1)
#     # image = np.reshape(tmp_image[0], (1, 28, 28, 1))
#     # print("[Yitao] %s" % image[0].size)
#     # print(image[0].shape)
#     # print(image[0].dtype)
#     # print(label.shape)
#     # print(label.dtype)
#     request.inputs['images'].CopyFrom(
#         tf.contrib.util.make_tensor_proto(image[0], shape=[5, 28, 28, 1]))
#     # print("Bangbangbang")
#     tmp_result = stub.Predict(request, 10.0)  # 5 seconds
#     # print(tmp_result)
#     sys.stdout.write('.')
#     sys.stdout.flush()

#   end = time.time()
#   print('\nFinished!')
#   print('It takes %s sec to run %d images by using MNIST' % (str(end - start), num_tests))


def myFuncWarmUp(stub, i, test_data_set):
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'caffe_mnist'
  request.model_spec.signature_name = 'predict_images'

  batchSize = 1
  durationSum = 0.0
  runNum = 13

  # test_data_set = mnist_input_data.read_data_sets(FLAGS.work_dir).test

  for k in range(runNum):
    start = time.time()

    image, label = test_data_set.next_batch(batchSize)
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image[0], shape=[batchSize, 28, 28, 1]))

    tmp_result = stub.Predict(request, 10.0)  # 5 seconds

    end = time.time()
    duration = (end - start)
    print("it takes %s sec" % str(duration))

    if (k != 0 and k != 3 and k != 8):
      durationSum += duration

  print("[Warm up] on average, it takes %s sec to run a batch of %d images over %d runs" % (str(durationSum / (runNum - 3)), batchSize, (runNum - 3)))

def myFuncParallel(stub, i, test_data_set):
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'caffe_mnist'
  request.model_spec.signature_name = 'predict_images'

  batchSize = 1
  durationSum = 0.0
  runNum = 10

  # test_data_set = mnist_input_data.read_data_sets(FLAGS.work_dir).test

  for k in range(runNum):
    start = time.time()

    image, label = test_data_set.next_batch(batchSize)
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image[0], shape=[batchSize, 28, 28, 1]))

    tmp_result = stub.Predict(request, 10.0)  # 5 seconds

    end = time.time()
    duration = (end - start)
    print("[thread-%d-%d] it takes %s sec" % (i, k, str(duration)))

    if True:
      durationSum += duration

  print("[Parallel-thread-%d] on average, it takes %s sec to run a batch of %d images over %d runs" % (i, str(durationSum / runNum), batchSize, runNum))


def main(_):
  # if FLAGS.num_tests > 10000:
  #   print('num_tests should not be greater than 10k')
  #   return
  # if not FLAGS.server:
  #   print('please specify server host:port')
  #   return
  # do_inference(FLAGS.server, FLAGS.work_dir,
  #                           FLAGS.concurrency, FLAGS.num_tests)


  hostport = FLAGS.server
  work_dir = FLAGS.work_dir
  concurrency = FLAGS.concurrency
  num_tests = FLAGS.num_tests

  test_data_set = mnist_input_data.read_data_sets(work_dir).test

  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  myFuncWarmUp(stub, 0, test_data_set)

  num_tests = 0
  tPool = []
  for i in range(num_tests):
    tPool.append(threading.Thread(target = myFuncParallel, args = (stub, i, test_data_set)))

  start = time.time()
  for i in range(num_tests):
    t = tPool[i]
    t.start()
    # time.sleep(2.0)

  for i in range(num_tests):
    t = tPool[i]
    t.join()

  end = time.time()

  print('\nFinished!')
  print('[Parallel] The total running time to run %d concurrent jobs is %s' % (num_tests, str(end - start)))



if __name__ == '__main__':
  tf.app.run()