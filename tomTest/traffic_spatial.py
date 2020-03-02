import os
import time
import pickle
import sys
import threading

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import threading
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/')
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/')

from modules_actdet.data_reader import DataReader
from modules_traffic.traffic_yolo import TrafficYolo
from modules_traffic.tiny_yolo_voc import TinyYoloVoc
from modules_traffic.caffe_googlenet import CaffeGooglenet
from modules_traffic.caffe_resnet50 import CaffeResnet50
from modules_traffic.caffe_resnet101 import CaffeResnet101
from modules_traffic.caffe_resnet152 import CaffeResnet152
from modules_traffic.mobilenet import Mobilenet
from modules_traffic.inception import Inception

# ============ Video Input Modules ============
reader = DataReader()
reader.Setup("/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/indoor_two_ppl.avi")

# ============ Object Detection Modules ============
# object detection
yolo = TrafficYolo()
yolo.Setup()
yolotiny = TinyYoloVoc()
yolotiny.Setup()

# person classification
# caffegooglenet = CaffeGooglenet()
# caffegooglenet.Setup()

# car classification
cafferesnet50 = CaffeResnet50()
cafferesnet50.Setup()
# cafferesnet101 = CaffeResnet101()
# cafferesnet101.Setup()
cafferesnet152 = CaffeResnet152()
cafferesnet152.Setup()

# ???
mobilenet = Mobilenet()
mobilenet.Setup()
inception = Inception()
inception.Setup()

stime = None
etime = None

def findPreviousModule(route_table, measure_module):
  tmp = route_table.split('-')
  for i in range(len(tmp)):
    if (tmp[i] == measure_module):
      return tmp[i - 1]

def runFrame(measure_module, request_input, frame_id):
# def runFrame(measure_module, sess_id, frame_id, reader):
  # # get input
  # if (measure_module == "traffic_yolo" or measure_module == "traffic_tinyyolo" or measure_module == "traffic_mobilenet"):
  #   frame_info = "%s-%s" % (sess_id, frame_id)
  #   route_index = 0
  #   frame_data = reader.PostProcess()  
  #   request_input = dict()
  #   request_input['client_input'] = frame_data['img']
  #   request_input['frame_info'] = frame_info
  #   request_input['route_table'] = route_table
  #   request_input['route_index'] = route_index
  # else:
  #   pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/pickle_tmp/%s" % findPreviousModule(route_table, measure_module), str(frame_id).zfill(3))
  #   with open(pickle_input) as f:
  #     request_input = pickle.load(f)

  if (measure_module == "traffic_yolo"):
    module_instance = yolo
  elif (measure_module == "traffic_tinyyolo"):
    module_instance = yolotiny
  elif (measure_module == "traffic_resnet152"):
    module_instance = cafferesnet152
  elif (measure_module == "traffic_resnet50"):
    module_instance = cafferesnet50
  elif (measure_module == "traffic_inception"):
    module_instance = inception
  elif (measure_module == "traffic_mobilenet"):
    module_instance = mobilenet

  start = time.time()

  module_instance.PreProcess(request_input = request_input, istub = istub, grpc_flag = False)
  module_instance.Apply()
  next_request = module_instance.PostProcess(grpc_flag = False)

  end = time.time()

  print("Finished frame %d for module %s in %.6f sec" % (frame_id, measure_module, float(end - start)))

  if (frame_id == 10):
    global stime
    stime = time.time()
  elif (frame_id == total_frame - 1):
    global etime
    etime = time.time()




ichannel = grpc.insecure_channel("localhost:8500")
istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

# real route table will be {traffic_yolo, traffic_tinyyolo} - {traffic_inception, traffic_mobilenet}
#                                              \ {traffic_resnet152, traffic_resnet50}

simple_route_table = "traffic_yolo-traffic_inception"
measure_module = "traffic_yolo"
route_table = simple_route_table

sess_id = "chain_traffic-000"
frame_id = 0

duration_sum = 0.0

input_fps = int(sys.argv[1])
total_frame = 120

if True:
  # get input
  if (measure_module == "traffic_yolo" or measure_module == "traffic_tinyyolo"):
    frame_info = "%s-%s" % (sess_id, "32")
    route_index = 0
    frame_data = reader.PostProcess()  
    request_input = dict()
    request_input['client_input'] = frame_data['img']
    request_input['frame_info'] = frame_info
    request_input['route_table'] = route_table
    request_input['route_index'] = route_index
  else:
    pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/pickle_tmp/%s" % findPreviousModule(route_table, measure_module), str(32).zfill(3))
    with open(pickle_input) as f:
      request_input = pickle.load(f)

while frame_id < total_frame:
  # frame_thread = threading.Thread(target = runFrame, args = (measure_module, sess_id, frame_id, reader,))
  frame_thread = threading.Thread(target = runFrame, args = (measure_module, request_input, frame_id,))
  frame_thread.start()

  time.sleep(1.0/input_fps)
  frame_id += 1

try:
  while True:
    time.sleep(60 * 60 * 24)
except KeyboardInterrupt:
  print("\nEnd by keyboard interrupt")
  print("<%f, %f> = %f over %d frames with fps of %f" % (float(stime), float(etime), float(etime) - float(stime), total_frame, (total_frame - 1 - 10) / (float(etime) - float(stime))))
