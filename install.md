```

sudo apt update -y
sudo apt upgrade -y
sudo rpi-update 

sudo apt install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev qt4-dev-tools python3-opencv python3-tk libatlas-base-dev python3-protobuf protobuf-compiler

sudo pip3 install tensorflow
sudo apt-get install python3-matplotlib python3-pil python3-lxml

mkdir ~/tensorflow1
cd ~/tensorflow1
git clone --recurse-submodules https://github.com/tensorflow/models.git

cd ~/tensorflow1/models/research
protoc object_detection/protos/*.proto --python_out=.

cd ~/tensorflow1/models/research/object_detection

## COCO-trained models

#wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz && tar -xzvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
#wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz && tar -xzvf ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz
#wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz && tar -xzvf ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz
#wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz && tar -xzvf ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz
#wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz && tar -xzvf ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz
#wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz && tar -xzvf ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
#wget http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz && tar -xzvf ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
#wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz && tar -xzvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
#wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz && tar -xzvf ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
#Prefered
wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz && tar -xzvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
#wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz && tar -xzvf ssd_inception_v2_coco_2018_01_28.tar.gz
#wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz && tar -xzvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
#wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz && tar -xzvf faster_rcnn_resnet50_coco_2018_01_28.tar.gz
#wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz && tar -xzvf faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz
#wget http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_2018_01_28.tar.gz && tar -xzvf rfcn_resnet101_coco_2018_01_28.tar.gz
#wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz && tar -xzvf faster_rcnn_resnet101_coco_2018_01_28.tar.gz
#wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz && tar -xzvf faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz
#wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz && tar -xzvf faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz
#wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz && tar -xzvf faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz
#wget http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz && tar -xzvf faster_rcnn_nas_coco_2018_01_28.tar.gz
#wget http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_lowproposals_coco_2018_01_28.tar.gz && tar -xzvf faster_rcnn_nas_lowproposals_coco_2018_01_28.tar.gz
#wget http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz && tar -xzvf mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz 
#wget http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz && tar -xzvf mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
#wget http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet101_atrous_coco_2018_01_28.tar.gz && tar -xzvf mask_rcnn_resnet101_atrous_coco_2018_01_28.tar.gz
#wget http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz && tar -xzvf mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz

## Kitti-trained models
#wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_kitti_2018_01_28.tar.gz && tar -xzvf faster_rcnn_resnet101_kitti_2018_01_28.tar.gz

## Open Images-trained models
#wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28.tar.gz && tar -xzvf faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28.tar.gz
#wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28.tar.gz && tar -xzvf faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28.tar.gz
#wget http://download.tensorflow.org/models/object_detection/facessd_mobilenet_v2_quantized_320x320_open_image_v4.tar.gz && tar -xzvf facessd_mobilenet_v2_quantized_320x320_open_image_v4.tar.gz
#wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz && tar -xzvf faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz
#wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_oid_v4_2018_12_12.tar.gz && tar -xzvf ssd_mobilenet_v2_oid_v4_2018_12_12.tar.gz
#wget http://download.tensorflow.org/models/object_detection/ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync_2019_01_20.tar.gz && tar -xzvf ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync_2019_01_20.tar.gz

## iNaturalist Species-trained models
#wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_fgvc_2018_07_19.tar.gz && tar -xzvf faster_rcnn_resnet101_fgvc_2018_07_19.tar.gz
#wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_fgvc_2018_07_19.tar.gz && tar -xzvf faster_rcnn_resnet50_fgvc_2018_07_19.tar.gz

## AVA v2.1 trained models
#wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_ava_v2.1_2018_04_30.tar.gz && tar -xzvf faster_rcnn_resnet101_ava_v2.1_2018_04_30.tar.gz



echo "export PYTHONPATH=$PYTHONPATH:~/tensorflow1/models/research:~/tensorflow1/models/research/slim" >> ~/.bashrc
export PYTHONPATH=$PYTHONPATH:~/tensorflow1/models/research:~/tensorflow1/models/research/slim

rm Object_detection_picamera.py
rm obj-config.ini
wget https://raw.githubusercontent.com/RattyDAVE/pi4b-object-detection/master/Object_detection_picamera.py
wget https://raw.githubusercontent.com/RattyDAVE/pi4b-object-detection/master/obj-config.ini

#---------------------- To Run ----------------------


cd ~/tensorflow1/models/research/object_detection
#python3 Object_detection_picamera.py 
#python3 Object_detection_picamera.py --usbcam

