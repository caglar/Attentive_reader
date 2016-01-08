import cPickle as pkl
import argparse

parser = argparse.ArgumentParser("To binarize the text files.")
parser.add_argument("dict",
        #"Input dictionary to binarize the text according to.",
                    type=argparse.FileType("r"))
parser.add_argument("inp",
        #           "Input text file to binarize the text according to.",
                    type=argparse.FileType("r"))
parser.add_argument("out",
        #            "Output pickle file.",
                    type=argparse.FileType("w"))

args = parser.parse_args()

inp_dict = pkl.load(args.dict)
inp_lines = args.inp.readlines()
out_toks = []

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

for line in inp_lines:
    toks = line.split(" ")
    bin_toks = replace_tokens(toks)
    if len(bin_toks) > 1:
        out_toks.append(bin_toks)

print "Writing into the file."
pkl.dump(out_toks, args.out)
