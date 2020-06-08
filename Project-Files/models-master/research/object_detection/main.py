import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2

from utils import label_map_util
from utils import visualization_utils as vis_util

from datetime import datetime
from dateutil import tz
import time
sys.path.append("..")

from database import *

def coco_detector(detection_graph):
    loc_input = int(input('Enter current location:\n1- Kitchen\n2- Back Door\n3- Front Door\n4- Living Room\n'))
    locations = {1: "Kitchen", 2: "Back Door", 3: "Front Door", 4: "Living Room"}
    try:
      location = locations[loc_input]
    except:
      print("Invalid location")
      print("Exiting Program")
      sys.exit()
    
    cap = cv2.VideoCapture(0) 
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        #Initialize database
        db = Database()
        #Time zone calculation
        HERE = tz.tzlocal()
        UTC = tz.gettz('UTC')
        #Boolean declarations
        initialization = False
        last_detected_local = 'Not detected'
        diff = 0
        detected = False
        while True:
          ret, image_np = cap.read()
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')

          # Actual detection.
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          

          #Initialize current time for first iteration
          if not initialization:
            if last_detected_local == 'Not detected':
              currTime = time.time()
              initialization = True
          
          # Label map maps 1 to person. Person must be the main content of the video capture. Prediction accuracy must be greater than 75%.
          # Calculate time in the frame
          if classes[0][0] == 1 and scores[0][0] > 0.50:
              diff = round(time.time() - currTime, 3)
              #print("Time detected {}".format(diff))
          else:
              # If already detected, write to the database for how long object has been detected
              if detected:
                try:
                  action = 'INSERT INTO record ' \
                          '(location, duration_detected, item_detected, time_recognized_utc, time_recognized_local) '\
                            'VALUES(%s, %s, %s, %s, %s);'
                  params = (location, diff, 'person', last_detected_utc, last_detected_local)
                  db.query(action, params)
                  print('Successfully written to databse!')
                except Exception as e:
                  print(e)
                  print('Failed to write to database')
                  #do something about it

              # Reset times for the next search
              currTime = time.time()
              diff = 0
              detected = False

          # Do something if object detected for more than 10 seconds
          if diff > 10:
            if not detected:
                last_detected_utc = datetime.utcnow()
                last_detected_local = last_detected_utc.replace(tzinfo=UTC).astimezone(HERE).strftime('%Y-%m-%d %H:%M:%SZ')
                print('Detected for 10 seconds, do something')
                
                
                detected = True
              
          
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)


          fps = cap.get(cv2.CAP_PROP_FPS)

          # Draw FPS
          cv2.putText(image_np,"FPS: {0:.2f}".format(fps),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),4,cv2.LINE_AA)

          # Draw seen data
          cv2.putText(image_np,'Detected for: ' + str(diff) + ' seconds',(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),4,cv2.LINE_AA)
          cv2.putText(image_np,'Last recognized: ' + str(last_detected_local),(10,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),4,cv2.LINE_AA)

          #Draw frame
          cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
          if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break



if __name__ == '__main__':
    ''' Using pretrained model
    MODEL_NAME = 'inference_graph'
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = 'training/labelmap.pbtxt'
    '''
    # What model to download.
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that are used to add a correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

    NUM_CLASSES = 90

    '''
    #Loading model into memory
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
      with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    
    # Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    coco_detector(detection_graph)