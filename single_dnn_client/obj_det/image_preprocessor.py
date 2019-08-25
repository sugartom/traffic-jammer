################################################################################
#  Helper functions  for image preoricessing                                   #
################################################################################
__author__ = "Alex Punnen"
__date__  = "March 2019"

from timeit import default_timer as timer
import numpy as np 
import cv2

def decode_image_opencv(img_path,max_height=800,swapRB=True,imagenet_mean = (0,0,0)):
  ### Going to create image vector via OpenCV
  #todo https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/getting%20started.html
  start = timer()
  image = cv2.imread(img_path,1)
  # print("original image shape=",image.shape)
  (h, w) = image.shape[:2]
  # print("Scale factor=", h/max_height) #we are currenly drawing in original so this is not relevant
  image = image_resize(image,height=max_height)
  org  = image
  #IMAGENET_MEAN = (103.939, 116.779, 123.68)
  # for certain architectues this is subtracted
  # please check/test your NW 
  # more details https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
  #IMAGENET_MEAN = (103.939, 116.779, 123.68)
  image = cv2.dnn.blobFromImage(image, scalefactor=1.0,mean=imagenet_mean, swapRB=swapRB)
  # this gives   shape as  (1, 3, 480, 640))
  image = np.transpose(image, (0, 2, 3, 1))
  # we get it after transpose as ('Input shape=', (1, 480, 640, 3))
  # print("resized image shape=",image.shape)
  # for original image we take the first image, (the first dim is number of images)
  #org = image[0,:,:,:] 
  #print("Draw shape=",org.shape)
  end = timer()
  # print("decode time=",end - start)
  return image,org

#https://stackoverflow.com/a/44659589/429476
# It is important to resize without loosing the aspect ratio for 
# good detection
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
  

def create_dummy_image(width=1067,height=800,channels=3):
    #Create a random numpy array
    return np.random.rand(height,width, channels).astype('f')

################################################################################
# Other functions just for illustration 
################################################################################
"""
def decode_image_tf_reader(img_path,max_height=480.0):
      #mage = tf.cast(img_tensor, tf.float32)
  
  #image = tf.image.resize_images(img_tensor, [800,1067])
  
  smallest_side = max_height # will losse some info
  height, width = tf.shape(image)[0], tf.shape(image)[1]
  height = tf.to_float(height)
  width = tf.to_float(width)
  
  scale = tf.cond(tf.greater(height, width),
                          lambda: smallest_side / width,
                          lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)

  image = tf.image.resize_images(image, [new_height, new_width])

  #image = tf.image.resize_images(image, [800,1200])

  #https://forums.fast.ai/t/how-is-vgg16-mean-calculated/4577/19
  VGG_MEAN = [123.68, 116.78, 103.94] # This is R-G-B for Imagenet
  #means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
  image = image - means 
  # swap to BGR    
  img_channel_swap = image[..., ::-1]
  image = tf.reverse(image, axis=[-1])
  #without the above preprocessing there is a miss of detection and change in weight
  image = tf.Session().run(image)
  #image = image[:, :, [2,1,0]] # swap channel from RGB to BGR
  end = timer()
  print("decode time=",end - start)
  return image,image
  """

# image = create_dummy_image()
# print(image)





# image, org = decode_image_opencv('/home/yitao/Documents/tf-1.2/TF-Serving-Downloads/dog.jpg')

# print(image.shape)

# batch_size = 5

# input = image 
# inputs = input
# for _ in range(batch_size-1):
#   inputs = np.append(inputs, input, axis=0)

# print(inputs.shape)