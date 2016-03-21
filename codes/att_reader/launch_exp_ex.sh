#!/bin/bash -e

DIR=~/stor/attentive_reader_simpler/codes/
SDIR=$DIR/att_reader/
MDIR=/u/cgulceh/stor/models
export PYTHONPATH=~/stor/attentive_reader_simpler/codes/:$PYTHONPATH


THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=0.85 python -u -O ${SDIR}/train_attentive_reader.py \
    --use_dq_sims 1 --use_desc_skip_c_g 0 --dim 280 --learn_h0 1 --lr 8e-5 --truncate -1 \
    --model "full_bptt_pruned2_lstm_uni_top4.npz" --batch_size 24 \
    --optimizer "adam" --validFreq 1000 --model_dir $MDIR --use_bidir 1
