import numpy as np
from scipy import stats
import os
import math
import argparse
from collections import Counter
from itertools import count
from time import time
from random import choice
import os, pickle
import nltk


parser = argparse.ArgumentParser(description='Word Similarity',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--target', required=True,
    help='word embeddings text file')

args = parser.parse_args()

def cosine_similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / n1 / n2

def levenshtein_edit(s1, s2):
    return nltk.edit_distance(s1,s2)

_, ext = os.path.splitext(args.target)
if ext in (".txt", ) :
    vocab, emb = [], []
    for i, line in zip(count(1), open(args.target)) :
        ss = line.split()
        vocab.append(ss[0])
        emb.append([float(x) for x in ss[1:]])
        if i % 10000 == 0 :
            logging.info('{} lines loaded'.format(i))
elif ext in (".pickle", ".pkl") :
    vocab, emb = pickle.load(open(args.target, 'rb'))
else :
    raise ValueError('Unsupported target vector file extent: {}'.format(args.target))
emb = np.array(emb)

print("suicide x sewerslide similarity:")
print(cosine_similarity(emb[0],emb[1]))
print("suicide x sewerslide levenshtein edit:", levenshtein_edit("suicide","sewerslide"))

print("suicide x sewerslidel similarity:")
print(cosine_similarity(emb[0],emb[2]))
print("suicide x sewerslidel levenshtein edit:", levenshtein_edit("suicide","sewerslidel"))

print("suicide x sucde similarity:")
print(cosine_similarity(emb[0],emb[3]))
print("suicide x sucde levenshtein edit:", levenshtein_edit("suicide","sucde"))

print("suicide x sulclde similarity:")
print(cosine_similarity(emb[0],emb[3]))
print("suicide x sulclde levenshtein edit:", levenshtein_edit("suicide","sulclde"))
##TO DO: 
##1 – get suciide variants next file 
##2 – create a text file of all words in post corpus WITH SUICIDE VARIANTS EXPUNGED
##3 – create a df w col 1 -> word we cross suicide w, col 2 --> whether its variant or nah, col 3 --> cosine sim, col 4 --> levenshtein edit
##5 - run logistic regression?? Or ig calculate if there's a stat sig difference b/w...
#           ...means of levenshtein edit for VAR GROUP and NON VAR GROUP
#           ...means of cosine similarity for VAR GROUP and NON VAR GROUP
##6 – another thing we can do is look for mroe MH/SM specific word vectors to train on