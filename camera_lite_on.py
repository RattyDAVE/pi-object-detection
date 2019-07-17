import os
import cv2
import numpy as np
import tensorflow as tf
import sys

#from https://www.tensorflow.org/lite/guide/hosted_models
#wget http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128_quant.tgz
PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'
TF_MODEL='mobilenet_v1_0.25_128_quant.tflite'
#TF_MODEL='mobilenet_v2_1.0_224.tflite'

NUM_CLASSES = 90

frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

camera = cv2.VideoCapture(0)
ret = camera.set(3,640)
ret = camera.set(4,480)

from utils import label_map_util
from utils import visualization_utils as vis_util

#from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

interpreter = tf.lite.Interpreter(model_path=TF_MODEL)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# check the type of the input tensor
floating_model = False
if input_details[0]['dtype'] == np.float32:
  floating_model = True

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2] 

print("height=",height)
print("width=",width)

#Loop it!
while(True):

        #FPS start counter
        t1 = cv2.getTickCount()
        
        #Get frame from camera
        ret, frame = camera.read()
        
        #Resize frame
        frame2 = cv2.resize(frame, (height,width))
        frame_expanded = np.expand_dims(frame2, axis=0)

        if floating_model:        
            frame_expanded = (np.float32(frame_expanded) - 127.5) / 127.5

        #Actual detection.     
        interpreter.set_tensor(input_details[0]['index'], frame_expanded)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        num = int(interpreter.get_tensor(output_details[3]['index'])[0])


        #Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=0.20)

        #Put the FPS over on the frame
        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
        
        #Show the Frame
        cv2.imshow('Object detector', frame)

        #FPS End counter
        t2 = cv2.getTickCount()
        
        #Work out FPS
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
