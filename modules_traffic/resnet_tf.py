import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import cv2

# sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/models/research/object_detection')
# from utils import label_map_util
# from utils import visualization_utils as vis_util

# import matplotlib.pyplot as plt
# PATH_TO_LABELS = os.path.join('/home/yitao/Documents/fun-project/tensorflow-related/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# IMAGE_SIZE = (12, 8)

class Resnet:
  def Setup(self):

    MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
    PATH_TO_FROZEN_GRAPH = '/home/yitao/Downloads/tmp/docker-share/module_traffic/models/%s/frozen_inference_graph.pb' % MODEL_NAME

    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
      config = tf.ConfigProto()
      config.gpu_options.per_process_gpu_memory_fraction = 0.3
      self.sess = tf.Session(config = config)
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      self.tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in self.tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        self.tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

  def PreProcess(self, input):
    self.input = np.expand_dims(input, axis=0)

  def Apply(self):
    # Run inference
    self.output_dict = self.sess.run(self.tensor_dict,
                           feed_dict={self.image_tensor: self.input})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    self.output_dict['num_detections'] = int(self.output_dict['num_detections'][0])
    self.output_dict['detection_classes'] = self.output_dict[
        'detection_classes'][0].astype(np.int64)
    self.output_dict['detection_boxes'] = self.output_dict['detection_boxes'][0]
    self.output_dict['detection_scores'] = self.output_dict['detection_scores'][0]
    if 'detection_masks' in self.output_dict:
      self.output_dict['detection_masks'] = self.output_dict['detection_masks'][0]

    # print(self.output_dict)

  def PostProcess(self):
    return self.output_dict

unit_test = False
if (unit_test):
  myInception = Inception()
  myInception.Setup()

  image_np = cv2.imread("/home/yitao/Documents/fun-project/tensorflow-related/models/research/object_detection/test_images/image1.jpg")
  # image_np_expanded = np.expand_dims(image_np, axis=0)
  myInception.PreProcess(image_np)
  myInception.Apply()
  output_dict = myInception.PostProcess()

  print(output_dict)

# vis_util.visualize_boxes_and_labels_on_image_array(
#       image_np,
#       output_dict['detection_boxes'],
#       output_dict['detection_classes'],
#       output_dict['detection_scores'],
#       category_index,
#       instance_masks=output_dict.get('detection_masks'),
#       use_normalized_coordinates=True,
#       line_thickness=8)
# plt.figure(figsize=IMAGE_SIZE)
# plt.imshow(image_np)
# plt.show()
# plt.savefig('tmp.png')
