#!/bin/bash

wget https://nvidia.box.com/shared/static/ncgzus5o23uck9i5oth2n8n06k340l6k.whl -O torch-1.4.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libfreetype6-dev python3-cffi libssl-dev libffi-dev

virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements_jetson.txt
pip3 install Cython
pip3 install numpy torch-1.4.0-cp36-cp36m-linux_aarch64.whl torchvision-0.2.2.post3-py2.py3-none-any.whl

mkdir answer
mkdir embedding
mkdir log
mkdir model
mkdir output
mkdir result