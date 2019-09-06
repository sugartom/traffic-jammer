import time
import numpy as np
import tensorflow as tf

from vgg16 import vgg16
from vgg16 import utils

class VGG16:
  def Setup(self):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    self.sess = tf.Session(config = config)
    self.vgg = vgg16.Vgg16()
    self.images = tf.placeholder("float", [1, 224, 224, 3])
    with tf.name_scope("content_vgg"):
      self.vgg.build(self.images)

  def PreProcess(self, input):
    self.input = input.reshape((1, 224, 224, 3))

  def Apply(self):
    self.prob = self.sess.run(self.vgg.prob, feed_dict = {self.images : self.input})
    # print(prob)

  def PostProcess(self):
    # output = utils.print_prob(self.prob, 'vgg16/synset.txt')
    # print(output)
    return self.prob

# myvgg = VGG16()
# myvgg.Setup()

# img1 = utils.load_image("vgg16/tiger.jpeg")
# myvgg.PreProcess(img1)
# myvgg.Apply()

# for i in range(10):
#   start = time.time()
#   myvgg.PreProcess(img1)
#   myvgg.Apply()
#   myvgg.PostProcess()
#   end = time.time()
#   print(end - start)



# img1 = utils.load_image("vgg16/tiger.jpeg")
# img2 = utils.load_image("vgg16/puzzle.jpeg")

# batch1 = img1.reshape((1, 224, 224, 3))
# batch2 = img2.reshape((1, 224, 224, 3))

# # batch = np.concatenate((batch1, batch2), 0)
# batch = np.concatenate((batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, 
#                         batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, 
#                         batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, 
#                         batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, 
#                         batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, 
#                         batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, 
#                         batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, 
#                         batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, 
#                         batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, 
#                         batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2, batch1, batch2), 0)

# # with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
# # with tf.device('/cpu:0'):
# with tf.Session() as sess:
#   images = tf.placeholder("float", [100, 224, 224, 3])
#   feed_dict = {images: batch}

#   vgg = vgg16.Vgg16()
#   with tf.name_scope("content_vgg"):
#     vgg.build(images)

#     prob = sess.run(vgg.prob, feed_dict=feed_dict)
#     print(prob)
#     utils.print_prob(prob[0], 'vgg16/synset.txt')
#     utils.print_prob(prob[1], 'vgg/synset.txt')
#     utils.print_prob(prob[2], 'vgg/synset.txt')
#     utils.print_prob(prob[3], 'vgg/synset.txt')
