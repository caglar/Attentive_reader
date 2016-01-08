#!/bin/bash -x
FNAME=$!
python clean_data.py --source_dir ./$FNAME/ --dest cleaned_$FNAME/
