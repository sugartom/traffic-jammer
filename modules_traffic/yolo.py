# Darkflow should be installed from: https://github.com/thtrieu/darkflow
from darkflow.net.build import TFNet
import numpy as np
import tensorflow as tf
import time
from modules_actdet.data_reader import DataReader
from modules_actdet.data_writer import DataWriter
from os.path import join 
import os 
import cv2
import sys

# # Yitao-TLS-Begin
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflow.python.framework import tensor_util
# # Yitao-TLS-End

# Place your downloaded cfg and weights under "checkpoints/"
YOLO_CONFIG = '/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/cfg'
YOLO_MODEL = '/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/cfg/yolo.cfg'
YOLO_WEIGHTS = '/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/checkpoints/yolo/yolo.weights'

GPU_ID = 0
GPU_UTIL = 0.5
YOLO_THRES = 0.4
YOLO_PEOPLE_LABEL = 'person'

'''
Input: {'img':img_np_array, 'meta':{'frame_id':frame_id}}

Output: {'img': img_np_array, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,x1,y1],
                        'label': label,
                        'conf': conf_score
                        }]
                }
        }
'''
class YOLO:

    @staticmethod
    def Setup():
        opt = { "config": YOLO_CONFIG,  
                "model": YOLO_MODEL, 
                "load": YOLO_WEIGHTS, 
                # "gpuName": GPU_ID,
                # "gpu": GPU_UTIL,
                "threshold": YOLO_THRES
            }
        YOLO.tfnet = TFNet(opt)

    def PreProcess(self, request_input, istub, grpc_flag):
        if (grpc_flag):
            self.request_input = str(tensor_util.MakeNdarray(request_input.inputs["client_input"]))
            self.image = cv2.imdecode(np.fromstring(self.request_input, dtype = np.uint8), -1)
        else:
            self.image = request_input['client_input']

        # print("[%.6f] debug" % time.time())
        # self.start = time.time()

        self.istub = istub

    def Apply(self):
        # self.start = time.time()
        # print("[@@@] dtype = %s, shape = %s" % (self.image.dtype, str(self.image.shape)))
        self.dets = YOLO.tfnet.return_predict(self.image, self.istub)
        # print("[@@@] This duration = %s" % str(time.time() - self.start))

        output = ""
        for d in self.dets:
            if d['label'] != YOLO_PEOPLE_LABEL:
                continue
            output += "%s|%s|%s|%s|%s|%s-" % (str(d['topleft']['x']), str(d['topleft']['y']), str(d['bottomright']['x']), str(d['bottomright']['y']), str(d['confidence']), str(d['label']))

        self.output = output[:-1]

    def PostProcess(self, grpc_flag):        
        if (grpc_flag):
            try:
                self.request_input
            except AttributeError:
                self.request_input = cv2.imencode('.jpg', self.image)[1].tostring()
                
            next_request = predict_pb2.PredictRequest()
            next_request.inputs['client_input'].CopyFrom(
              tf.make_tensor_proto(self.request_input))
            next_request.inputs['objdet_output'].CopyFrom(
              tf.make_tensor_proto(self.output))
            return next_request
        else:
            result = dict()
            result['client_input'] = self.image
            result['objdet_output'] = self.output
            return result
