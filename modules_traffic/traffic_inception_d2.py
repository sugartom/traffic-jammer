import cv2
import numpy as np
import os

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2

import string_int_label_map_pb2 as labelmap
from google.protobuf import text_format

INCEPTION_THRES = 0.4
INCEPTION_PEOPLE_LABEL = "person"

from tensorflow.python.framework import tensor_util
import tensorflow as tf

class TrafficInception:

  # initialize static variable here
  @staticmethod
  def Setup():
    s = open('%s/modules_traffic/mscoco_complete_label_map.pbtxt' % os.environ['TRAFFIC_JAMMER_PATH'], 'r').read()
    mymap = labelmap.StringIntLabelMap()
    TrafficInception._label_map = text_format.Parse(s, mymap)

  def decode_image_opencv(self, image, max_height = 800, swapRB = True, imagenet_mean = (0, 0, 0)):
    # image = cv2.imread(img_path, 1)
    (h, w) = image.shape[:2]
    image = self.image_resize(image, height = max_height)
    org  = image
    image = cv2.dnn.blobFromImage(image, scalefactor=1.0, mean = imagenet_mean, swapRB = swapRB)
    image = np.transpose(image, (0, 2, 3, 1))
    return image, org

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

  def box_normal_to_pixel(self, box, dim, scalefactor = 1):
    # https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py
    height, width = dim[0], dim[1]
    ymin = int(box[0] * height * scalefactor)
    xmin = int(box[1] * width * scalefactor)

    ymax = int(box[2] * height * scalefactor)
    xmax= int(box[3] * width * scalefactor)
    return np.array([xmin, ymin, xmax, ymax])

  def get_label(self, index):
    return TrafficInception._label_map.item[index].display_name

  # convert predict_pb2.PredictRequest()'s content to data_dict
  # input: request["image"] = image
  #        request["meta"] = meta
  # output: data_dict["image"] = image
  #         data_dict["meta"] = meta
  def GetDataDict(self, request, grpc_flag):
    data_dict = dict()

    # do the conversion for each key in predict_pb2.PredictRequest()
    if (grpc_flag):
      raw_image = tensor_util.MakeNdarray(request.inputs["raw_image"])
      output_flag = int(tensor_util.MakeNdarray(request.inputs["output_flag"]))
    else:
      raw_image = request["raw_image"]
      output_flag = request["output_flag"]

    image, org = self.decode_image_opencv(raw_image)
    image = image.astype(np.uint8)
    data_dict["client_input"] = image
    data_dict["original_shape"] = org.shape
    # data_dict["raw_image"] = raw_image
    data_dict["output_flag"] = output_flag

    return data_dict

  # for an array of requests from a batch, convert them to a dict,
  # where each key has a lit of values
  # input: data_array = [{"image": image1, "meta": meta1}, {"image": image2, "meta": meta2}]
  # output: batched_data_dict = {"image": [image1, image2], "meta": [meta1, meta2]}
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

      # original_shape
      batched_data_dict["original_shape"] = []
      for data in data_array:
        batched_data_dict["original_shape"].append(data["original_shape"])

      # # raw_image
      # batched_data_dict["raw_image"] = []
      # for data in data_array:
      #   batched_data_dict["raw_image"].append(data["raw_image"])

      # output_flag
      batched_data_dict["output_flag"] = []
      for data in data_array:
        batched_data_dict["output_flag"].append(data["output_flag"])

      return batched_data_dict

  # input: batched_data_dict = {"image": [image1, image2], "meta": [meta1, meta2]}
  # output: batched_result_dict = {"bounding_boxes": [[bb1_in_image1, bb2_in_image1], [bb1_in_image2]]}
  def Apply(self, batched_data_dict, batch_size, istub):
    if (batch_size != len(batched_data_dict[batched_data_dict.keys()[0]])):
      print("[Error] Apply() batch size not matched...")
      return None
    else:
      batched_result_dict = dict()

      batched_input = batched_data_dict["client_input"]

      request = predict_pb2.PredictRequest()
      request.model_spec.name = 'traffic_inception'
      request.model_spec.signature_name = 'serving_default'
      request.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(batched_input, shape = batched_input.shape))

      result = istub.Predict(request, 10.0)

      boxes = tf.make_ndarray(result.outputs['detection_boxes'])
      scores = tf.make_ndarray(result.outputs['detection_scores'])
      labels = tf.make_ndarray(result.outputs['detection_classes'])

      batched_result_dict["boxes"] = boxes
      batched_result_dict["scores"] = scores
      batched_result_dict["labels"] = labels
      batched_result_dict["original_shape"] = batched_data_dict["original_shape"]
      # batched_result_dict["raw_image"] = batched_data_dict["raw_image"]
      batched_result_dict["output_flag"] = batched_data_dict["output_flag"]

      # for i in range(batch_size):
      #   output = ""
      #   dim = batched_data_dict["original_shape"][i]
      #   for box, score, label in zip(boxes[i], scores[i], labels[i]):
      #     if score < INCEPTION_THRES:
      #       break
      #     box = self.box_normal_to_pixel(box, dim)
      #     b = box.astype(int)
      #     class_label = self.get_label(int(label))
      #     if (class_label == INCEPTION_PEOPLE_LABEL):
      #       output += "%s|%s|%s|%s|%s|%s-" % (str(b[0]), str(b[1]), str(b[2]), str(b[3]), str(score), str(class_label))
      #   otuput = output[:-1]

      #   batched_result_dict[""]

      return batched_result_dict

  # input: batched_result_dict = {"bounding_boxes": [[bb1_in_image1, bb2_in_image1], [bb1_in_image2]]}
  # output: batched_result_array = [{"bounding_boxes": [bb1_in_image1, bb2_in_image1]}, {"bounding_boxes": [bb1_in_image2]}]
  def GetBatchedResultArray(self, batched_result_dict, batch_size):
    if (batch_size != len(batched_result_dict[batched_result_dict.keys()[0]])):
      print("[Error] GetBatchedResultArray() batch size not matched...")
      return None
    else:
      batched_result_array = []

      for i in range(batch_size):
        my_dict = dict()

        output_flag = batched_result_dict["output_flag"][i]
        if (output_flag == 1):
          output = ""
          dim = batched_result_dict["original_shape"][i]
          for box, score, label in zip(batched_result_dict["boxes"][i], batched_result_dict["scores"][i], batched_result_dict["labels"][i]):
            if score < INCEPTION_THRES:
              break
            box = self.box_normal_to_pixel(box, dim)
            b = box.astype(int)
            class_label = self.get_label(int(label))
            if (class_label == INCEPTION_PEOPLE_LABEL):
              output += "%s|%s|%s|%s|%s|%s-" % (str(b[0]), str(b[1]), str(b[2]), str(b[3]), str(score), str(class_label))
          output = output[:-1]
        else:
          output = ""

        my_dict["objdet_output"] = [output]
        # my_dict["raw_image"] = [batched_result_dict["raw_image"][i]]
        batched_result_array.append(my_dict)

      return batched_result_array

  # input: result_dict = {"bounding_boxes": [bb1_in_image1, bb2_in_image1]}
  # output: result_list = [{"bounding_boxes": bb1_in_image1}, {"bounding_boxes": bb2_in_image1}]
  def GetResultList(self, result_dict):
    result_list = []
    for i in range(len(result_dict[result_dict.keys()[0]])):
      if (result_dict["objdet_output"][i] != ""):
        # result_list.append({"objdet_output": result_dict["objdet_output"][i], "raw_image": result_dict["raw_image"][i]})
        result_list.append({"objdet_output": result_dict["objdet_output"][i]})
    return result_list

  # input: result = {"bounding_boxes": bb1_in_image1}
  # output: next_request["boudning_boxes"] = bb1_in_image1
  def GetNextRequest(self, result, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      # next_request.inputs['raw_image'].CopyFrom(
      #   tf.make_tensor_proto(result["raw_image"]))
      next_request.inputs["objdet_output"].CopyFrom(
        tf.make_tensor_proto(result["objdet_output"]))
    else:
      next_request = dict()
      # next_request["raw_image"] = result["raw_image"]
      next_request["objdet_output"] = result["objdet_output"]
    return next_request
