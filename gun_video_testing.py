import time
start = time.time()
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
 
import cv2
 
sys.path.append("..")
from object_detection.utils import ops as utils_ops
 
if StrictVersion(tf.__version__) < StrictVersion('1.4.0'):
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
 
from utils import label_map_util
from utils import visualization_utils as vis_util
 
MODEL_NAME = 'knife_detection'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
 
PATH_TO_LABELS = os.path.join('data', 'gun.pbtxt')
 
NUM_CLASSES = 6
 
def detect_in_video():
  
    out = cv2.VideoWriter('gun7.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1280, 720))
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
            
   
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
 
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
           
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
           
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
           
            cap = cv2.VideoCapture('~\\models\\research\\object_detection\\test_images1\\video7.mp4')
 
            while(cap.isOpened()):
                
                ret, frame = cap.read()
 
                detect_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
                image_np_expanded = np.expand_dims(detect_frame, axis=0)
 
               
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores,
                        detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
 
             
                vis_util.visualize_boxes_and_labels_on_image_array(
                    detect_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=.20)
 
             
                output_rgb = cv2.cvtColor(detect_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('detect_output', output_rgb)
                out.write(output_rgb)
               
              
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
           
            out.release()
            cap.release()
      
 
def main():
    detect_in_video()
 
if __name__ =='__main__':
    main()
    
end =  time.time()
print("Execution Time: ", end - start)
