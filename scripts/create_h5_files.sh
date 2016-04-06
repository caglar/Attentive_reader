#!/bin/bash -e

DIR=/data/lisatmp4/gulcehrc/reading_comprehension_data/cleaned_cnn/
echo "creating the test file."
MODE='test'
NEWS_SOURCE="cnn"
TOUT_FILE=${DIR}/${NEWS_SOURCE}_${MODE}_data.h5 
SOURCE_PATH=${DIR}/$MODE/
python create_h5_files_all.py --input_dir ${SOURCE_PATH} --vocab ${DIR}/cleaned_cnn_vocab.pkl --output_file ${TOUT_FILE}

echo "creating the validation file."
MODE='validation'
TOUT_FILE=${DIR}/${NEWS_SOURCE}_${MODE}_data.h5
SOURCE_PATH=${DIR}/$MODE/
python create_h5_files_all.py --input_dir ${SOURCE_PATH} --vocab ${DIR}/cleaned_cnn_vocab.pkl --output_file ${TOUT_FILE}

echo "creating the training file."
MODE='training'
TOUT_FILE=${DIR}/${NEWS_SOURCE}_${MODE}_data.h5 
SOURCE_PATH=${DIR}/$MODE/
python create_h5_files_all.py --input_dir ${SOURCE_PATH} --vocab ${DIR}/cleaned_cnn_vocab.pkl --output_file ${TOUT_FILE}

exit;
