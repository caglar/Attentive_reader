"""

This script parses the news files and at each line, just leaves three texts.
The first line is the text.
The second line is the question.
The last line is the answer.

"""
import os
import argparse

parser = argparse.ArgumentParser("Parse the RC texts and clean them.")
parser.add_argument('--source_dir')
parser.add_argument('--dest_dir')

args = parser.parse_args()

assert args.source_dir is not None and os.path.exists(args.source_dir)
assert args.dest_dir is not None

if not os.path.exists(args.dest_dir):
    print "creating the directory %s " % args.dest_dir
    os.makedirs(args.dest_dir)

print "Started cleaning the files."

for root, dir_, files in os.walk(args.source_dir):
    for file_ in files:
        rt_fil = os.path.join(args.source_dir, file_)
        source_fil = open(rt_fil, 'r')
        lines = source_fil.readlines()
        dataset = lines[2:8]
        first_part = file_.split(".question")[0]

        dest_fil = os.path.join(args.dest_dir, "{}_cleaned.question".format(first_part))

        to_fil = open(dest_fil, 'w')
        for line in dataset:
            line = line.strip('\n')
            if line:
                print >>to_fil, line
        to_fil.close()

print "Cleaning the data is completed."
