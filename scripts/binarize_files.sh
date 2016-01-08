#!/bin/bash -x

SDIR=/u/yyu/stor/cnn/pureNoUnify/att8/
python binarize_text2.py  $SDIR/dict_v8.pkl $SDIR/train8p $SDIR/train8q $SDIR/train8a --out_dir $SDIR --out_prefix train8v2
python binarize_text2.py  $SDIR/dict_v8.pkl $SDIR/valid8p $SDIR/valid8q $SDIR/valid8a --out_dir $SDIR --out_prefix valid8v2
