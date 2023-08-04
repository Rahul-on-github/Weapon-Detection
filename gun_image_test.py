import time
start = time.time()
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd

if StrictVersion(tf.__version__) < StrictVersion('1.4.0'):
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
  
os.chdir('~\\models\\research\\object_detection')
  
sys.path.append("..")


from object_detection.utils import label_map_util


from object_detection.utils import visualization_utils as vis_util



MODEL_NAME = 'knife_detection'




PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('data', 'gun.pbtxt')

NUM_CLASSES = 6



'''
#Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
'''   
    
    
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')    
    
    
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


PATH_TO_TEST_IMAGES_DIR = '~\\models\\research\\object_detection\\test_images\\'
os.chdir(PATH_TO_TEST_IMAGES_DIR)
TEST_IMAGE_DIRS = os.listdir(PATH_TO_TEST_IMAGES_DIR)

IMAGE_SIZE = (12, 8)

output_image_path = ('~\\models\\research\\object_detection\\gun_output\\pics\\')
output_csv_path = ('~\\models\\research\\object_detection\\gun_output\\csv\\')

for image_folder in TEST_IMAGE_DIRS:
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        TEST_IMAGE_PATHS = os.listdir(os.path.join(image_folder))
        os.makedirs(output_image_path+image_folder)
        data = pd.DataFrame()
        for image_path in TEST_IMAGE_PATHS:
          image = Image.open(image_folder + '//'+image_path)
          width, height = image.size
          image_np = load_image_into_numpy_array(image)
          image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
          image_np_expanded = np.expand_dims(image_np, axis=0)
          (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
       
          
          cv2.imwrite(output_image_path+image_folder+'\\'+image_path.split('\\')[-1],image_np)
          
          s_boxes = boxes[scores > 0.5]
          s_classes = classes[scores > 0.5]
          s_scores=scores[scores>0.5]
          
         
          for i in range(len(s_classes)):

              newdata= pd.DataFrame(0, index=range(1), columns=range(7))
              newdata.iloc[0,0] = image_path.split("\\")[-1].split('.')[0]
              newdata.iloc[0,1] = s_boxes[i][0]*height  
              newdata.iloc[0,2] = s_boxes[i][1]*width     
              newdata.iloc[0,3] = s_boxes[i][2]*height   
              newdata.iloc[0,4] = s_boxes[i][3]*width   
              newdata.iloc[0,5] = s_scores[i]
              newdata.iloc[0,6] = s_classes[i]
    
              data = data.append(newdata)
          data.to_csv(output_csv_path+image_folder+'.csv',index = False)
      
end =  time.time()
print("Execution Time: ", end - start)
