# Darkflow should be installed from: https://github.com/thtrieu/darkflow
from darkflow.net.build import TFNet
import numpy as np
from time import time 
# from modules_actdet.data_reader import DataReader
# from modules_actdet.data_writer import DataWriter
from os.path import join 
import os 

# Place your downloaded cfg and weights under "checkpoints/"
YOLO_CONFIG = '/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/cfg'
YOLO_MODEL = '/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/cfg/yolo.cfg'
YOLO_WEIGHTS = '/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/bin/yolo.weights'

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
    dets = []
    input = {}

    def Setup(self):
        opt = { "config": YOLO_CONFIG,  
                "model": YOLO_MODEL, 
                "load": YOLO_WEIGHTS, 
                "gpuName": GPU_ID,
                "gpu": GPU_UTIL,
                "threshold": YOLO_THRES
            }
        self.tfnet = TFNet(opt)

    def PreProcess(self, input):
        self.input = input 


    def Apply(self):
        if self.input:
            self.dets = self.tfnet.return_predict(self.input['img'])   

    def PostProcess(self):
        # output = self.input
        # if not self.input:
        #     return output 

        # output['meta']['obj'] = []
        # for d in self.dets:
        #     if d['label'] != YOLO_PEOPLE_LABEL:
        #         continue 
        #     output['meta']['obj'].append({'box':[int(d['topleft']['x']), int(d['topleft']['y']),
        #                                         int(d['bottomright']['x']), int(d['bottomright']['y'])],
        #                                         'label': d['label'],
        #                                         'conf': d['confidence']})
        output = self.input
        return output
