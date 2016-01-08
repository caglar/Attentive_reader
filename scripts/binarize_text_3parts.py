import cPickle as pkl
import argparse

parser = argparse.ArgumentParser("To binarize the text files.")
parser.add_argument("dict",
        #"Input dictionary to binarize the text according to.",
                    type=argparse.FileType("r"))

parser.add_argument("passage",
        #           "Input text file to binarize the text according to.",
                    type=argparse.FileType("r"))

parser.add_argument("question",
        #           "Input text file to binarize the text according to.",
                    type=argparse.FileType("r"))

parser.add_argument("answer",
        #           "Input text file to binarize the text according to.",
                    type=argparse.FileType("r"))

parser.add_argument("--out_dir",
                    type=str)

parser.add_argument("--out_prefix",
                    type=str)


args = parser.parse_args()

inp_dict = pkl.load(args.dict)
passage_lines = args.passage.readlines()
question_lines = args.question.readlines()
answer_lines = args.answer.readlines()


#replace_tokens = lambda ws: [inp_dict[w.strip()] if w.strip() \
#        in inp_dict and w.strip() =! '' and w.strip() != "" and w.strip() is not None else inp_dict['UNK'] for w in ws]

def replace_tokens(seq):
    new_seq = []
    for w in seq:
        nw = w.strip()
        if nw is not None and nw != "" and nw != '':
            if nw in inp_dict:
                new_seq.append(inp_dict[nw])
            else:
                new_seq.append(inp_dict['UNK'])

    return new_seq

out_pass, out_qs, out_ans = [], [], []

for p, q, a in zip(passage_lines, question_lines, answer_lines):
    ptoks = p.split(" ")
    bin_ptoks = replace_tokens(ptoks) + [inp_dict['EOS']]
    qtoks = q.split(" ")
    bin_qtoks = replace_tokens(qtoks) + [inp_dict['EOS']]
    atoks = a.split(" ")
    bin_atoks = replace_tokens(atoks)
    if len(bin_ptoks) > 1 and len(bin_qtoks) > 1 and len(bin_atoks) >= 1:
        out_pass.append(bin_ptoks)
        out_qs.append(bin_qtoks)
        out_ans.append(bin_atoks)

prfx = args.out_dir + args.out_prefix
print "Writing into the file."
pkl.dump(out_pass, open(prfx + "_pass.pkl", "w"))
pkl.dump(out_qs, open(prfx + "_qs.pkl", "w"))
pkl.dump(out_ans, open(prfx + "_ans.pkl", "w"))
