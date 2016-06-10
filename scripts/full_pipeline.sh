#!/bin/bash -x


NEWS_SOURCE="cnn"
DIR=/data/lisatmp4/gulcehrc/reading_comprehension_data/${NEWS_SOURCE}
MAX_VOCAB_SIZE=60000

VOCAB_NAME=clean_${NEWS_SOURCE}_vsize_${MAX_VOCAB_SIZE}.pkl
VFILE=${DIR}/${VOCAB_NAME}

preprocess_file() {
    MODE=$1
    echo "$MODE"
    echo "cleaning the ${MODE} file."
    ORIG_DIR=${DIR}/questions/${MODE}
    CLEAN_OUT_DIR=${DIR}/cleaned_${MODE}/

    mkdir -p ${CLEAN_OUT_DIR}
    python clean_data.py --source_dir ${ORIG_DIR} --dest ${CLEAN_OUT_DIR}
    echo "saved the cleaned ${MODE} file to ${CLEAN_OUT_DIR}."
    echo

    if [ "${MODE}" == "validation" ]
    then
        echo "creating the vocab"
        python preprocess.py --source_file_dir ${CLEAN_OUT_DIR} --out_file_name ${VFILE} --max_vocab_size ${MAX_VOCAB_SIZE}
    fi

    if [ -e ${VFILE} ]
    then
        echo "creating the ${MODE} file."
        TOUT_FILE=${DIR}/${NEWS_SOURCE}_${MODE}_${MAX_VOCAB_SIZE}_data.h5
        python create_h5_files_all.py --input_dir ${CLEAN_OUT_DIR} --vocab ${VFILE} --output_file ${TOUT_FILE}
        echo "Successfully created the ${MODE} file."
    else
        echo "ERROR: Vocabulary file does not exist" 2>&1
        exit 1;
    fi
}

preprocess_file training
preprocess_file validation
preprocess_file test

exit;
