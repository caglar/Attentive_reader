#!/bin/bash -x

NEWS_SOURCE="cnn"
DIR=/u/yyu/stor/caglar/rc-data/cnn

TRAIN_SOURCE_PATH=/u/cgulceh/data/rc-data/${NEWS_SOURCE}/questions/cleaned_training/
VALID_SOURCE_PATH=/u/cgulceh/data/rc-data/${NEWS_SOURCE}/questions/cleaned_validation/
TEST_SOURCE_PATH=/u/cgulceh/data/rc-data/${NEWS_SOURCE}/questions/cleaned_test/

MODE="train"
QVOCAB=${DIR}/q_vocab_${NEWS_SOURCE}_${MODE}.pkl
AVOCAB=${DIR}/a_vocab_${NEWS_SOURCE}_${MODE}.pkl
DVOCAB=${DIR}/d_vocab_${NEWS_SOURCE}_${MODE}.pkl

# Create the vocabularies for the train source files first!
#python preprocess.py --source_file_dir ${TRAIN_SOURCE_PATH} --vocab_type q --out_file_name ${QVOCAB}
#python preprocess.py --source_file_dir ${TRAIN_SOURCE_PATH} --vocab_type a --out_file_name ${AVOCAB}
#python preprocess.py --source_file_dir ${TRAIN_SOURCE_PATH} --vocab_type d --out_file_name ${DVOCAB}

#Create the h5 file for the train data
#OUT_FILE=${DIR}/${NEWS_SOURCE}_${MODE}_data.h5
#python create_h5_files.py --input_dir ${TRAIN_SOURCE_PATH} --vocab_desc ${DVOCAB} --vocab_ans ${AVOCAB} --vocab_q ${QVOCAB} --output_file ${OUT_FILE}

#MODE='valid'
#VOUT_FILE=${DIR}/${NEWS_SOURCE}_${MODE}_data.h5
#python -m ipdb create_h5_files.py --input_dir ${VALID_SOURCE_PATH} --vocab_desc ${DVOCAB} --vocab_ans ${AVOCAB} --vocab_q ${QVOCAB} --output_file ${VOUT_FILE}
MODE='test'
TOUT_FILE=${DIR}/${NEWS_SOURCE}_${MODE}_data.h5
python -m ipdb create_h5_files_sents.py --input_dir ${TEST_SOURCE_PATH} --vocab_desc ${DVOCAB} --vocab_ans ${AVOCAB} --vocab_q ${QVOCAB} --output_file ${TOUT_FILE}


exit;
