#!/bin/bash

sudo apt-get update
sudo apt-get install -y libglfw3-dev libglfw3
sudo apt-get install libassimp-dev
pip install --user --pre --upgrade PyOpenGL PyOpenGL_accelerate
pip install --user cython
pip install --user cyglfw3
pip install --user pyassimp==3.3
pip install --user imgaug

cd /data/AugmentedAutoencoder
pip install --user .

echo 'AugmentedAutoencoder installed. Enjoy your module.'
