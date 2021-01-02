import threading
import cv2
import grpc
import time
import numpy as np
import os
import pickle
import sys

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow_serving.apis import prediction_service_pb2_grpc

sys.path.append('/home/yitao/Documents/edge/D2-system/')
from utils_d2 import misc
from modules_d2.video_reader import VideoReader

sys.path.append(os.environ['TRAFFIC_JAMMER_PATH'])
from modules_traffic.traffic_yolo_d2 import TrafficYolo
from modules_traffic.traffic_inception_d2 import TrafficInception
from modules_traffic.traffic_resnet_d2 import TrafficResnet
from modules_traffic.traffic_mobilenet_d2 import TrafficMobilenet

TrafficYolo.Setup()
TrafficInception.Setup()
TrafficResnet.Setup()
TrafficMobilenet.Setup()

ichannel = grpc.insecure_channel("localhost:8500")
istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

module_name = "traffic_yolo"
# module_name = "traffic_inception"
# module_name = "traffic_resnet"
# module_name = "traffic_mobilenet"

pickle_directory = "%s/pickle_d2/traffic-jammer/%s" % (os.environ['RIM_DOCKER_SHARE'], module_name)
if not os.path.exists(pickle_directory):
  os.makedirs(pickle_directory)

batch_size = 1
parallel_level = 1
run_num = 500

def runBatch(batch_size, run_num, tid):
  start = time.time()

  reader = VideoReader()
  # reader.Setup("%s/indoor_2min.mp4" % os.environ['CAESAR_EDGE_PATH'])
  # reader.Setup("/home/yitao/Downloads/2020-11-07-17_26_41/2_Pike_NS_Seattle.mp4")
  # reader.Setup("/home/yitao/Downloads/2020-11-07-17_26_41/catsmeow2_NewOrleans.mp4")
  # reader.Setup("/home/yitao/Downloads/2020-11-07-17_26_41/tsstreet_NYC.mp4")
  # reader.Setup("/home/yitao/Downloads/2020-11-07-17_26_41/5thAve_PineSt_Seattle.mp4")
  reader.Setup("/home/yitao/Downloads/2020-11-07-17_26_41/hollywoodblvd_LA.mp4")

  frame_id = 0
  batch_id = 0

  while (batch_id < run_num):
    module_instance = misc.prepareModuleInstance(module_name)
    data_array = []

    if (module_name == "traffic_yolo"):
      for i in range(batch_size):
        client_input = reader.PostProcess()
        request = dict()
        request["client_input"] = client_input
        data_dict = module_instance.GetDataDict(request, grpc_flag = False)
        data_array.append(data_dict)
        frame_id += 1
    elif (module_name == "traffic_resnet" or module_name == "traffic_mobilenet" or module_name == "traffic_inception"):
      pickle_input = "%s/%s" % ("%s/pickle_d2/%s/%s" % (os.environ['RIM_DOCKER_SHARE'], "traffic-jammer", "traffic_inception"), str(1).zfill(3))
      request = pickle.load(open(pickle_input))
      data_dict = module_instance.GetDataDict(request, grpc_flag = False)
      for i in range(batch_size):
        data_array.append(data_dict)

    batched_data_dict = module_instance.GetBatchedDataDict(data_array, batch_size)

    batched_result_dict = module_instance.Apply(batched_data_dict, batch_size, istub)

    batched_result_array = module_instance.GetBatchedResultArray(batched_result_dict, batch_size)

    for i in range(len(batched_result_array)):
      # deal with the outputs of the ith input in the batch
      result_dict = batched_result_array[i]

      # each input might have more than one outputs
      result_list = module_instance.GetResultList(result_dict)

      for result in result_list:
        next_request = module_instance.GetNextRequest(result, grpc_flag = False)

        if (module_name == "traffic_yolo"):
          print(len(next_request["objdet_output"].split("-")))
        #   pickle_output = "%s/%s" % (pickle_directory, str(frame_id).zfill(3))
        #   with open(pickle_output, 'w') as f:
        #     pickle.dump(next_request, f)

      if (len(result_list) == 0):
        print(0)


    batch_id += 1

  end = time.time()
  print("[Thread-%d] it takes %.3f sec to run %d batches of batch size %d" % (tid, end - start, run_num, batch_size))


# ========================================================================================================================

start = time.time()

thread_pool = []
for i in range(parallel_level):
  t = threading.Thread(target = runBatch, args = (batch_size, run_num, i))
  thread_pool.append(t)
  t.start()

for t in thread_pool:
  t.join()

end = time.time()
print("overall time = %.3f sec" % (end - start))
