from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.test.utils import common_texts, get_tmpfile, datapath
# import pandas as pd
import numpy as np
import os
import csv
import time 
# # visualization libraries
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import matplotlib
# import matplotlib.pyplot as plt


###################################
# Run gensim word2vec model on khan academy problem attempt dataset
# Input: space-delimited problem attempt dataset converted to tokens with concatenated (exercise,problem_type,correctness)  
# Output: the model generates the embedding vectors representing each token
#	In addition, for each (exercise,problem_type), we output the difference vector between True and False
#	and then find the problem type that most closely approximates the difference vector space 

# Validation: the validation data-set will be xercise name with human-entered prerequisites (sometimes>1) 
# 	accuracy rate will be measured as the % of exercises (out of those with prereqs available), where the predicted 
#	this file does attempt to validate the output. That will occur later.    


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
    

def write_token_file(file_name, data_array):
    '''write token file. Estimated time: 1.5 sec for 1M rows'''
    path = os.path.expanduser('~/output/'+file_name+'.csv')
    print(path)
    open_file = open(path, "w")
    with open_file:
        [open_file.write(data + '\n') for data in data_array]
        
# def write_token_file(file_name, data_array):
#     '''write file. Estimated time: 1.5 sec for 1M rows'''
#     # [TODO]: can we reduce runtime if we don't use csvwriter and just write file with open? 
#     path = os.path.expanduser('~/output/'+file_name+'.csv')
#     print(path)
#     open_file = open(path, "w")
#     with open_file:
#         csvwriter = csv.writer(open_file)
#         for row in data_array:
#             csvwriter.writerows(row)
        
def write_vector_file(file_name, data_array):
    '''write the embedding vectors. Estimated time: 1.5 sec for 1M rows'''
    path = os.path.expanduser('~/output/'+file_name+'.out')
    print(path)
    np.savetxt(path, data_array, delimiter = ',')


start = time.time() 
read_filename = os.path.expanduser('~/sorted_data/tokenize_data_sorted.csv')
skills = read_data(read_filename)

#############################################
## Model the skills data and create the embedding representation 
# Input: the sorted skill tokens for each learning session
# Output: embedding vector for each token and the index with the name of token in same order
## Time: For 1M rows of data, this takes 
window_size = 10 
embed_size = 30
iter_num = 30

model = Word2Vec(skills, size = embed_size, window = window_size, iter =iter_num)
vectors = model.wv.vectors
vector_index = model.wv.index2word
print(vector_index)

# store the vectors and the vector index locally 
write_vector_file('embed_vectors_full', vectors)
write_token_file('embed_index_full', vector_index)

end = time.time()
print(end-start)

# small dataset using gensim approach to 
# using the manual read dataset: 35 sec 
# using gensim approach: 50 sec 







