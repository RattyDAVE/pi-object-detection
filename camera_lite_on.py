#from https://www.tensorflow.org/lite/guide/hosted_models
#wget http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128_quant.tgz

import tensorflow as tf
import cv2
import numpy as np
from utils import label_map_util
from utils import visualization_utils as vis_util
import configparser

config = configparser.ConfigParser()
config.read('camera_lite_on.ini')

IM_WIDTH = int(config['DEFAULT']['IM_WIDTH'])
IM_HEIGHT= int(config['DEFAULT']['IM_HEIGHT'])
MODEL_NAME = config['DEFAULT']['MODEL_NAME']

CWD_PATH = os.getcwd()
PATH_TO_LABELS = os.path.join(CWD_PATH,'data',config['DEFAULT']['PATH_TO_LABELS'])
NUM_CLASSES = int(config['DEFAULT']['NUM_CLASSES'])

#a = tf.lite.Interpreter('mobilenet_v1_1.0_224.tflite')
#a = tf.lite.Interpreter('mobilenet_v2_1.0_224.tflite')
a = tf.lite.Interpreter(MODEL_NAME)

labels = load_labels(args.label_file)
#label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#category_index = label_map_util.create_category_index(categories)

a.allocate_tensors()
input_details = a.get_input_details()
output_details = a.get_output_details()

frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

camera = cv2.VideoCapture(0)
ret = camera.set(3,IM_WIDTH)
ret = camera.set(4,IM_HEIGHT)

while(True):

    #FPS start counter
    t1 = cv2.getTickCount()
        
    #Get frame from camera
    ret, frame = camera.read()

    #frame = cv2.imread("test.jpg")
    frame_expanded = cv2.resize(frame, (300,300))
    frame_expanded = np.expand_dims(frame_expanded, axis=0)
    #frame_expanded = (np.float32(frame_expanded) - 127.5) / 127.5
    frame_expanded = np.uint8(np.float32(frame_expanded))

    #Actual detection.
    a.set_tensor(input_details[0]['index'], frame_expanded)
    a.invoke()

    boxes = a.get_tensor(output_details[0]['index']) 
    classes = a.get_tensor(output_details[1]['index'])
    scores = a.get_tensor(output_details[2]['index'])
    num = a.get_tensor(output_details[3]['index'])
  
    #Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            labels,
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

    cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()
