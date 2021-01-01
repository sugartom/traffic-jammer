# Darkflow should be installed from: https://github.com/sugartom/darkflow
from darkflow.net.build import TFNet
import numpy as np

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

  def GetDataDict(self, request, grpc_flag):
    data_dict = dict()

    # do the conversion for each key in predict_pb2.PredictRequest()
    if (grpc_flag):
      raw_image = tensor_util.MakeNdarray(request.inputs["client_input"])
    else:
      raw_image = request["client_input"]

    image = TrafficYolo.tfnet.framework.resize_input(raw_image)
    image = np.expand_dims(image, 0)

    data_dict["client_input"] = image
    data_dict["raw_image"] = raw_image

    return data_dict

  def GetBatchedDataDict(self, data_array, batch_size):
    if (len(data_array) != batch_size):
      print("[Error] GetBatchedDataDict() batch size not matched...")
      return None
    else:
      batched_data_dict = dict()

      # for each key in data_array[0], convert it to batched_data_dict[key][]
      batched_data_dict["client_input"] = data_array[0]["client_input"]
      for data in data_array[1:]:
        batched_data_dict["client_input"] = np.append(batched_data_dict["client_input"], data["client_input"], axis = 0)

      # raw_image
      batched_data_dict["raw_image"] = []
      for data in data_array:
        batched_data_dict["raw_image"].append(data["raw_image"])

      return batched_data_dict

  def Apply(self, batched_data_dict, batch_size, istub):
    if (batch_size != len(batched_data_dict[batched_data_dict.keys()[0]])):
      print("[Error] Apply() batch size not matched...")
      return None
    else:
      batched_result_dict = dict()

      batched_input = batched_data_dict["client_input"]

      batched_result = TrafficYolo.tfnet.return_batched_predict(batched_input, "traffic_yolo", istub)

      batched_result_dict["batched_result"] = batched_result
      batched_result_dict["raw_image"] = batched_data_dict["raw_image"]

      return batched_result_dict

  def GetBatchedResultArray(self, batched_result_dict, batch_size):
    if (batch_size != len(batched_result_dict[batched_result_dict.keys()[0]])):
      print("[Error] GetBatchedResultArray() batch size not matched...")
      return None
    else:
      batched_result_array = []

      for i in range(batch_size):
        my_dict = dict()

        dets = batched_result_dict["batched_result"][i]
        output = ""
        for d in dets:
          output += "%s|%s|%s|%s|%s|%s-" % (str(d['topleft']['x']), str(d['topleft']['y']), str(d['bottomright']['x']), str(d['bottomright']['y']), str(d['confidence']), str(d['label']))
        output = output[:-1]
        my_dict["objdet_output"] = [output]
        my_dict["raw_image"] = [batched_result_dict["raw_image"][i]]
        batched_result_array.append(my_dict)

      return batched_result_array

  # input: result_dict = {"bounding_boxes": [bb1_in_image1, bb2_in_image1]}
  # output: result_list = [{"bounding_boxes": bb1_in_image1}, {"bounding_boxes": bb2_in_image1}]
  def GetResultList(self, result_dict):
    result_list = []
    for i in range(len(result_dict[result_dict.keys()[0]])):
      if (result_dict["objdet_output"][i] != ""):
        result_list.append({"objdet_output": result_dict["objdet_output"][i], "raw_image": result_dict["raw_image"][i]})
    return result_list

  # input: result = {"bounding_boxes": bb1_in_image1}
  # output: next_request["boudning_boxes"] = bb1_in_image1
  def GetNextRequest(self, result, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      next_request.inputs['raw_image'].CopyFrom(
        tf.make_tensor_proto(result["raw_image"]))
      next_request.inputs["objdet_output"].CopyFrom(
        tf.make_tensor_proto(result["objdet_output"]))
    else:
      next_request = dict()
      next_request["raw_image"] = result["raw_image"]
      next_request["objdet_output"] = result["objdet_output"]
    return next_request
