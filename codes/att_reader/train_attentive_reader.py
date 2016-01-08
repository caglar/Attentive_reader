from attentive_reader import train
import argparse
import cPickle as pkl
import os
import pprint as pp
from utils import create_model_name, create_entity_mask
from pkl_data_iterator import get_data_files

import numpy


def main(job_id, params):
    train_files, valid_files, vpath = get_data_files(mode=params["data_mode"])
    dvocab = pkl.load(open(vpath))
    emb_sz = len(dvocab) + 1
    eyem = numpy.eye(emb_sz)

    sent_opts = {'use_sent_reps': params['use_sent_reps'],
                 'sent_end_tok_ids': [dvocab['.'],
                                      dvocab['!']]}
    cost_mask = create_entity_mask(dvocab, eyem)
    pp.pprint(params)
    new_model_name = create_model_name(params['model'], params)

    print "The model will be saved to %s" % new_model_name
    validerr, validcost = train(saveto=new_model_name,
                                reload_=params['reload'][0],
                                dim=params['dim'][0],
                                decay_c=params['decay-c'][0],
                                clip_c=params['clip-c'][0],
                                lrate=params['learning-rate'][0],
                                optimizer=params['optimizer'],
                                ms_nlayers=params['ms_nlayers'],
                                n_words_q=emb_sz,
                                n_words_desc=emb_sz,
                                n_words_ans=emb_sz,
                                cost_mask=cost_mask,
                                vocab=vpath,
                                use_elu_g=params['use_elu_g'],
                                encoder_desc=params['encoder_desc'],
                                encoder_desc_word=params['encoder_desc_word'],
                                encoder_q=params['encoder_q'],
                                debug=params['debug'],
                                dim_word_q=params['dim_word_q'][0],
                                dim_word_ans=params['dim_word_ans'][0],
                                dim_word_desc=params['dim_word_desc'][0],
                                dim_proj=params['dim_proj'][0],
                                valid_datasets=params['valid_datasets'],
                                batch_size=params['batch_size'],
                                model_dir=params['model_dir'],
                                learn_h0=params['learn_h0'],
                                use_dq_sims=params['use_dq_sims'],
                                use_desc_skip_c_g=params['use_desc_skip_c_g'],
                                truncate=params['truncate'],
                                use_bidir=params['use_bidir'],
                                valid_batch_size=params['batch_size'],
                                validFreq=params['validFreq'],
                                pkl_train_files=train_files,
                                pkl_valid_files=valid_files,
                                dispFreq=20,
                                eyem=eyem,
                                saveFreq=3000,
                                patience=1000,
                                use_dropout=params['use-dropout'][0],
                                **sent_opts)

    return validerr, validcost


if __name__ == '__main__':
    model_dir = "./"

    parser = argparse.ArgumentParser("The different variations of the attentive reader.")
    parser.add_argument("--use_dq_sims", default=1, type=int)
    parser.add_argument("--use_desc_skip_c_g", type=int, default=0)
    parser.add_argument("--truncate", default=50, type=int)
    parser.add_argument("--model", default="new_model.npz")
    parser.add_argument("--dim", default=250, type=int)
    parser.add_argument("--learn_h0", default=0, type=int)
    parser.add_argument("--model_dir", default=model_dir, type=str)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--validFreq", default=150, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--use_sent_reps", default=0, type=int)
    parser.add_argument("--unit_type", default="lstm", type=str)
    parser.add_argument("--clip_grad", default=10.0, type=float)
    parser.add_argument("--use_elu_g", default=0, type=int)
    parser.add_argument("--data_mode", default="top4", type=str)
    parser.add_argument("--debug", default=0, type=int)
    parser.add_argument("--use_bidir", default=0, type=int)
    parser.add_argument("--ms_nlayers", default=2, type=int)
    parser.add_argument("--reloadm", default=0, type=int)
    args = parser.parse_args()

    main(0, {
        'debug': args.debug,
        'model': [args.model],
        'dim': [int(args.dim)],
        'dim_word_q': [int(args.dim)],
        'dim_word_ans': [int(args.dim)],
        'dim_proj': [int(args.dim)],
        'dim_word_desc': [int(args.dim)],
        'use_dq_sims': args.use_dq_sims,
        'use_desc_skip_c_g': args.use_desc_skip_c_g,
        'valid_datasets': ['/u/yyu/stor/caglar/rc-data/cnn/cnn_valid_data2.h5',
                           '/u/yyu/stor/caglar/rc-data/cnn/cnn_valid_data2.h5'],
        'decay-c': [0.],
        'use_bidir': args.use_bidir,
        'ms_nlayers': args.ms_nlayers,
        'clip-c': [args.clip_grad],
        'use_elu_g': args.use_elu_g,
        'encoder_desc': args.unit_type,
        'encoder_desc_word': args.unit_type,
        'encoder_q': args.unit_type,
        'truncate': int(args.truncate),
        'learn_h0': args.learn_h0,
        'use-dropout': [True],
        'model_dir': args.model_dir,
        'optimizer': args.optimizer,
        'validFreq': args.validFreq,
        'data_mode': args.data_mode,
        'use_sent_reps': args.use_sent_reps,
        'learning-rate': [args.lr],
        'batch_size': args.batch_size,
        'reload': [args.reloadm]})
