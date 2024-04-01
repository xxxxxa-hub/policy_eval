#!/bin/sh

pip install tensorflow==2.6.0
conda install cudatoolkit cudnn
pip install -e .
