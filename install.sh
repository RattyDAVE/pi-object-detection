#!/bin/sh

cd ~

sudo apt update -y
#sudo apt upgrade -y

#On a new RPI4 this is a must as fixes firmware
#sudo rpi-update 

sudo apt install -y libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev qt4-dev-tools python3-opencv python-opencv python3-tk libatlas-base-dev python3-protobuf protobuf-compiler 
sudo apt install -y python3-matplotlib python3-pil python3-lxml python3-grpcio python3-keras-preprocessing python3-protobuf python3-keras-applications python3-wrapt python3-termcolor python3-astor python3-h5py python3-markdown


### Keep and eye on the below for a verion for the RPI 4. 
git clone https://github.com/PINTO0309/Tensorflow-bin.git
cd ~/Tensorflow-bin
pip3 install --upgrade setuptools
ls tensorflow-1*cp37*armv7l*
pip3 install tensorflow-1*cp37*armv7l*



##mv tensorflow-1.14.0-cp35-cp35m-linux_armv7l.whl tensorflow-1.14.0-cp37-cp37m-linux_armv7l.whl

#wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/tensorflow-1.14.0-cp37-cp37m-linux_armv7l.whl
#pip3 install --upgrade setuptools
#pip3 install tensorflow-1.14.0-cp37-cp37m-linux_armv7l.whl

#sudo pip3 install tensorflow

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


#Mobile models
#wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2019_08_14.tar.gz && tar -xzvf ssd_mobilenet_v3_large_coco_2019_08_14.tar.gz
#wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_small_coco_2019_08_14.tar.gz && tar -xzvf ssd_mobilenet_v3_small_coco_2019_08_14.tar.gz

#Pixel4 Edge TPU models
#wget https://storage.cloud.google.com/mobilenet_edgetpu/checkpoints/ssdlite_mobilenet_edgetpu_coco_quant.tar.gz  && tar -xzvf ssdlite_mobilenet_edgetpu_coco_quant.tar.gz

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

#Tensorflow LITE models
#wget http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128_quant.tgz && tar -xzvf mobilenet_v1_0.25_128_quant.tgz

echo "export PYTHONPATH=$PYTHONPATH:~/tensorflow1/models/research:~/tensorflow1/models/research/slim" >> ~/.bashrc
export PYTHONPATH=$PYTHONPATH:~/tensorflow1/models/research:~/tensorflow1/models/research/slim

rm camera_on.py
rm obj-config.ini
wget https://raw.githubusercontent.com/RattyDAVE/pi-object-detection/master/camera_on.py
wget https://raw.githubusercontent.com/RattyDAVE/pi-object-detection/master/obj-config.ini
