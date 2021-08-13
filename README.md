# Object Detection on Raspberry Pi 4/3

### Tested On

- RPi4 with USB camera
- RPi4 with Raspberry Pi Camera Module.
- RPi3 with USB camera
- RPi3 with Raspberry Pi Camera Module.

The purpose is to get the object detection and proof of concept working in the minimum time.

Ethos:
- We use pre-compiled binaries where possible from the Raspberry Pi repository.
- The python code contains the minimal needed to be functional. 

#### Defaults
- Image size of 640x480
- ssdlite_mobilenet_v2_coco_2018_05_09

#### Rate
- RPi4 You can expect 2.3 FP/S using the defaults.
- RPi3 You can expect 1.2 FP/S using the defaults.

This uses pretrained models and can has the ability to change the model easy using the configuration file.

## Scripts
### To install run the following.
- To install Tensorflow 1
```
curl https://raw.githubusercontent.com/RattyDAVE/pi-object-detection/master/install.sh|/bin/sh
```
- To install Tensorflow 2 beta
```
curl https://raw.githubusercontent.com/RattyDAVE/pi-object-detection/master/install2.sh|/bin/sh
```


### To start the object detetection run the following
```
curl https://raw.githubusercontent.com/RattyDAVE/pi-object-detection/master/run.sh|/bin/sh
```

### To uninstall and wipe all traces run the following.
```
curl https://raw.githubusercontent.com/RattyDAVE/pi-object-detection/master/uninstall.sh|/bin/sh
```



## Models

All models are taken from [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)

The default is ssdlite_mobilenet_v2_coco_2018_05_09. You can change the models but uncommenting the line in the [install.sh](https://github.com/RattyDAVE/pi4b-object-detection/blob/master/install.sh) and the [obj-config.ini](https://github.com/RattyDAVE/pi4b-object-detection/blob/master/obj-config.ini)

#### COCO-trained models from [COCO dataset](http://mscoco.org)
90 Classes

Size   |Model | Status rpi4| FPS rpi4
  --- | --- | --- | ---
73MB|ssd_mobilenet_v1_coco_2018_01_28
44MB|ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03
81MB|ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18
49MB|ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18
29MB|ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03
129MB|ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
349MB|ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
179MB|ssd_mobilenet_v2_coco_2018_03_29
138MB|ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03
48MB|*ssdlite_mobilenet_v2_coco_2018_05_09*|**WORKS**|2.8
265MB|ssd_inception_v2_coco_2018_01_28
142MB|faster_rcnn_inception_v2_coco_2018_01_28
363MB|faster_rcnn_resnet50_coco_2018_01_28
363MB|faster_rcnn_resnet50_lowproposals_coco_2018_01_28
622MB|rfcn_resnet101_coco_2018_01_28
565MB|faster_rcnn_resnet101_coco_2018_01_28
565MB|faster_rcnn_resnet101_lowproposals_coco_2018_01_28
641MB|faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28
641MB|faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28
1.09GB|faster_rcnn_nas_coco_2018_01_28
1.09GB|faster_rcnn_nas_lowproposals_coco_2018_01_28
693MB|mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28
169MB|mask_rcnn_inception_v2_coco_2018_01_28
631MB|mask_rcnn_resnet101_atrous_coco_2018_01_28
428MB|mask_rcnn_resnet50_atrous_coco_2018_01_28


### Mobile models

Size   |Model | Status rpi4| FPS rpi4
  --- | --- | --- | ---
? |ssd_mobilenet_v3_large_coco
? |ssd_mobilenet_v3_small_coco

### Pixel4 Edge TPU models

Size   |Model | Status rpi4| FPS rpi4
  --- | --- | --- | ---
?|ssd_mobilenet_edgetpu_coco

#### Kitti-trained models from [Kitti dataset](http://www.cvlibs.net/datasets/kitti/)
2 Classes

Size   |Model | Status rpi4| FPS rpi4
  --- | --- | --- | ---
555MB|faster_rcnn_resnet101_kitti_2018_01_28|**FAILED (bad alloc)**

#### Open Images-trained models from [Open Images dataset](https://github.com/openimages/dataset)
601 Classes

Size   |Model | Status rpi4| FPS 4
  --- | --- | --- | ---
680MB|faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28
680MB|faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28
124MB|facessd_mobilenet_v2_quantized_320x320_open_image_v4
682MB|faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12
151MB|ssd_mobilenet_v2_oid_v4_2018_12_12 | **Works** | 1.5
608MB|ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync_2019_01_20

#### iNaturalist Species-trained models from [iNaturalist Species Detection Dataset](https://github.com/visipedia/inat_comp/blob/master/2017/README.md#bounding-boxes)
2854 Classs

Size   |Model | Status rpi4| FPS rpi4
  --- | --- | --- | ---
868MB|faster_rcnn_resnet101_fgvc_2018_07_19
666MB|faster_rcnn_resnet50_fgvc_2018_07_19|**FAILED (bad alloc)**

#### AVA v2.1 trained models from [AVA v2.1 dataset](https://research.google.com/ava/)
AVA is a project that provides audiovisual annotations of video for improving our understanding of human activity.

90 Classes

Size   |Model | Status rpi4| FPS rpi4
  --- | --- | --- | ---
565MB|faster_rcnn_resnet101_ava_v2.1_2018_04_30|**FAILED (bad alloc)**


TypeError: int() argument must be a string, a bytes-like object or a number, not 'NoneType'

