#!/bin/bash -e

THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=0.85 python train_nmt.py 

