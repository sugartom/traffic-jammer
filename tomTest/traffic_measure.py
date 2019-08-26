import time

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import threading
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/')
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/')

from modules_actdet.data_reader import DataReader
from modules_traffic.yolo import Yolo
from modules_traffic.tiny_yolo_voc import TinyYoloVoc


# ============ Video Input Modules ============
reader = DataReader()
reader.Setup("/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/indoor_two_ppl.avi")

# ============ Object Detection Modules ============
yolo = Yolo()
yolo.Setup()
`
yolotiny = TinyYoloVoc()
yolotiny.Setup()




ichannel = grpc.insecure_channel("localhost:8500")
istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

# simple_route_table = "Yolo"
simple_route_table = "TinyYoloVoc"

route_table = simple_route_table

sess_id = "chain_traffic-000"
frame_id = 0

duration_sum = 0.0

while (frame_id < 160):
  start = time.time()
  
  frame_info = "%s-%s" % (sess_id, frame_id)

  route_index = 0

  frame_data = reader.PostProcess()
  if not frame_data:  # end of video 
    break
  
  request_input = dict()
  request_input['client_input'] = frame_data['img']
  request_input['frame_info'] = frame_info
  request_input['route_table'] = route_table
  request_input['route_index'] = route_index

  for i in range(len(route_table.split('-'))):
    current_model = route_table.split('-')[request_input['route_index']]

    if (current_model == "Yolo"):
      module_instance = Yolo()
    elif (current_model == "TinyYoloVoc"):
      module_instance = TinyYoloVoc()

    module_instance.PreProcess(request_input = request_input, istub = istub, grpc_flag = False)
    module_instance.Apply()
    # next_request = module_instance.PostProcess(grpc_flag = False)

    # next_request['frame_info'] = request_input['frame_info']
    # next_request['route_table'] = request_input['route_table']
    # next_request['route_index'] = request_input['route_index'] + 1

    # request_input = next_request

  end = time.time()
  duration = end - start
  duration_sum += duration
  print("duration = %s" % str(duration))

  frame_id += 1

print("On average, it takes %f sec per frame" % (duration_sum / 160))
