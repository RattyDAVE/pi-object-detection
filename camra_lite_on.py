import os
import cv2
import numpy as np
import tensorflow as tf
import sys


PATH_TO_LABELS = 'labels'
TF_MODEL='detect.tflite'

frame_rate_calc = 1
freq = cv2.getTickFrequency()

font = cv2.FONT_HERSHEY_SIMPLEX

camera = cv2.VideoCapture(0)
ret = camera.set(3,IM_WIDTH)
ret = camera.set(4,IM_HEIGHT)

from utils import label_map_util
from utils import visualization_utils as vis_util

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

interpreter = tf.contrib.lite.Interpreter(model_path=TF_MODEL)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

boxes = interpreter.get_tensor(output_details[0]['index'])
classes = interpreter.get_tensor(output_details[1]['index'])
sores = interpreter.get_tensor(output_details[2]['index'])
num = int(interpreter.get_tensor(output_details[3]['index'])[0])

#Loop it!
while(True):

        #FPS start counter
        t1 = cv2.getTickCount()
        
        #Get frame from camera
        ret, frame = camera.read()
        
        #Resize frame
        frame_expanded = np.expand_dims(frame, axis=0)
        
        # Actual detection.
        #(boxes, scores, classes, num) = sess.run(
        #    [detection_boxes, detection_scores, detection_classes, num_detections],
        #    feed_dict={image_tensor: frame_expanded})
        
        interpreter.set_tensor(input_details[0]['index'], frame_expanded)
        interpreter.invoke()

        # Visualization of the results of a detection.
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

