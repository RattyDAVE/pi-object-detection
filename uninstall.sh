#!/bin/sh

sudo pip3 uninstall $(pip3 show tensorflow | grep Requires | sed 's/Requires: //g; s/,//g')
sudo apt purge -y libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev qt4-dev-tools python3-opencv python3-tk libatlas-base-dev python3-protobuf protobuf-compiler python3-matplotlib python3-pil python3-lxml
rm -rf ~/tensorflow1
rm -rf ~/Tensorflow-bin
sed -i '/tensorflow1/d' ~/.bashrc
