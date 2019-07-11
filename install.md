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

echo "export PYTHONPATH=$PYTHONPATH:~/tensorflow1/models/research:~/tensorflow1/models/research/slim" >> ~/.bashrc

export PYTHONPATH=$PYTHONPATH:~/tensorflow1/models/research:~/tensorflow1/models/research/slim

cd ~/tensorflow1/models/research
protoc object_detection/protos/*.proto --python_out=.

cd ~/tensorflow1/models/research/object_detection
wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
tar -xzvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

wget https://raw.githubusercontent.com/RattyDAVE/pi4b-object-detection/master/Object_detection_picamera.py

#---------------------- To Run ----------------------


cd ~/tensorflow1/models/research/object_detection
#python3 Object_detection_picamera.py 
#python3 Object_detection_picamera.py --usbcam




======
sed '/\(^VAR5=\).*/ s//\1VALUE10/' sample.txt

MODEL_NAME =
PATH_TO_LABLES =
NUM_CLASSES =


```
