# Object Detection on Raspberry Pi 4
This has been tested on a Raspberry Pi 4 with a USB camera and the Raspberry Pi Camera Module.

The perpose is to get the object detection and proof of concept working in the minimum time.

Ethos:
- We use pre-compiled binaries where possible from the Raspberry Pi repository.
- The python code contains the minimal needed to be functional. 

#### Defaults
- Image size of 640x480
- ssdlite_mobilenet_v2_coco_2018_05_09

You can expect 2.3 FP/S using the defaults.

This uses pretrained models and can has the ability to change the model easy using the configuration file.

## Scripts
### To install run the following.
```
curl https://raw.githubusercontent.com/RattyDAVE/pi4b-object-detection/master/install.sh|/bin/sh
```

### To start the object detetection run the following
```
curl https://raw.githubusercontent.com/RattyDAVE/pi4b-object-detection/master/run.sh|/bin/sh
```

### To uninstall and wipe all traces run the following.
```
curl https://raw.githubusercontent.com/RattyDAVE/pi4b-object-detection/master/uninstall.sh|/bin/sh
```



## Models

All models are taken from [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

The default is ssdlite_mobilenet_v2_coco_2018_05_09. You can change the models but uncommenting the line in the [install.sh](https://github.com/RattyDAVE/pi4b-object-detection/blob/master/install.sh) and the [obj-config.ini](https://github.com/RattyDAVE/pi4b-object-detection/blob/master/obj-config.ini)

#### COCO-trained models
- ssd_mobilenet_v1_coco_2018_01_28
- ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03
- ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18
- ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18
- ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03
- ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
- ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
- ssd_mobilenet_v2_coco_2018_03_29
- ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03
- *ssdlite_mobilenet_v2_coco_2018_05_09*
- ssd_inception_v2_coco_2018_01_28
- faster_rcnn_inception_v2_coco_2018_01_28
- faster_rcnn_resnet50_coco_2018_01_28
- faster_rcnn_resnet50_lowproposals_coco_2018_01_28
- rfcn_resnet101_coco_2018_01_28
- faster_rcnn_resnet101_coco_2018_01_28
- faster_rcnn_resnet101_lowproposals_coco_2018_01_28
- faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28
- faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28
- faster_rcnn_nas_coco_2018_01_28
- faster_rcnn_nas_lowproposals_coco_2018_01_28
- mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28
- mask_rcnn_inception_v2_coco_2018_01_28
- mask_rcnn_resnet101_atrous_coco_2018_01_28
- mask_rcnn_resnet50_atrous_coco_2018_01_28

#### Kitti-trained models - NOT TESTED
- faster_rcnn_resnet101_kitti_2018_01_28

#### Open Images-trained models - NOT TESTED
- faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28
- faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28
- facessd_mobilenet_v2_quantized_320x320_open_image_v4
- faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12
- ssd_mobilenet_v2_oid_v4_2018_12_12
- ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync_2019_01_20

#### iNaturalist Species-trained models - NOT TESTED
- faster_rcnn_resnet101_fgvc_2018_07_19
- faster_rcnn_resnet50_fgvc_2018_07_19

#### AVA v2.1 trained models - NOT TESTED
- faster_rcnn_resnet101_ava_v2.1_2018_04_30
