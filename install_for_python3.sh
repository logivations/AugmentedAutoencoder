#!/bin/bash

sudo apt-get update
sudo apt-get install -y libglfw3-dev libglfw3
sudo apt-get install libassimp-dev
pip3 install --user --pre --upgrade PyOpenGL PyOpenGL_accelerate
pip3 install --user cython
pip3 install --user cyglfw3
pip3 install --user pyassimp==3.3
pip3 install --user imgaug

cd /data/AugmentedAutoencoder
pip3 uninstall -y auto-pose
pip3 install --user .

echo 'AugmentedAutoencoder installed. Enjoy your module.'
