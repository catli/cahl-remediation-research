from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.test.utils import common_texts, get_tmpfile, datapath
import pandas as pd
import numpy as np
import os
import csv
import time 
# visualization libraries
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt


###################################
# Run gensim word2vec model on khan academy problem attempt dataset
# Input: space-delimited problem attempt dataset converted to tokens with concatenated (exercise,problem_type,correctness)  
# Output: the model generates the embedding vectors representing each token
#	In addition, for each (exercise,problem_type), we output the difference vector between True and False
#	and then find the problem type that most closely approximates the difference vector space 
# Validation: the validation data-set will be th exercise name with human-entered prerequisites (sometimes>1) 
# 	accuracy rate will be measured as the % of exercises (out of those with prereqs available), where the predicted 
# 
# 




def read_data(read_filename):
    '''
        read tokenized data into arrays of tokens formatted for Word2Vec
        we could use gensim LineSentence method but that seems to take 40% longer
        to run
    '''
    reader = open(read_filename,'r')
    skills = []
    for line in reader:
        row = line.split(" ")
        row[-1] = row[-1].strip()
        skills.append(row)
    return skills
    







