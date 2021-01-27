#!/bin/bash

virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt

mkdir answer
mkdir embedding
mkdir log
mkdir model
mkdir output
mkdir result