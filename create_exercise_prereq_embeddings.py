'''
   iCreate input and output embedding array describing 
   the set of target exercise and prerequisites 
   the input will be the average embeddings for 
   the target exercise where the response is correct ('true')

   and the output will be the average embeddings for the 
   associated prerequisite exercise where the response is correct
   ('true')
'''
from util_functions import  read_prerequisite_data
from util_functions import write_token_file
from util_functions import write_vector_file
from util_functions import read_embedding_vectors
from util_functions import read_tokens
from create_similar_token_sets import generate_similarity_tokens 
import os
import numpy as np
import pdb

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import random 


def read_vectors_and_index(output_path, read_file_affix):
    # read the exercise embedding vectors
    # read the exercise embedding index
    vector_filepath = output_path + 'embed_vectors_' + read_file_affix 
    index_filepath = output_path + 'embed_index_' + read_file_affix 
    response_vectors = read_embedding_vectors(vector_filepath)
    response_tokens = read_tokens(index_filepath)
    prerequisites = read_prerequisite_data('prerequisites', is_json_file=False)
    return response_vectors, response_tokens, prerequisites



def create_input_output_vectors(response_vectors, response_tokens, prerequisites):
    # create input output vectors from the exercise embedding
    token_array = []
    for target_exercise in prerequisites:
        target_prereq = prerequisites[target_exercise][0]
        try: 
            target_index = response_tokens.index(target_exercise+'|true')
            prereq_index = response_tokens.index(target_prereq+'|true')
        except:
            continue
        if 'prereq_input_vectors' in locals():
            prereq_input_vectors = np.concatenate(
                (prereq_input_vectors, 
                np.array([ response_vectors[target_index]]) ), axis=0)
            prereq_output_vectors = np.concatenate(
                (prereq_output_vectors, 
                np.array([ response_vectors[prereq_index]]) ), axis=0)
        else:
            prereq_input_vectors = np.array([ response_vectors[target_index]])
            prereq_output_vectors = np.array([ response_vectors[prereq_index]])
        token_array.append((target_exercise, target_prereq))
    return  prereq_input_vectors, prereq_output_vectors, token_array




root_path = os.path.split(os.getcwd())[0] + '/'
output_path = root_path + 'cahl_output' + '/'
read_file_affix = 'exercise'
response_vectors, response_tokens, prerequisites = read_vectors_and_index(output_path, read_file_affix)
prereq_input_vectors, prereq_output_vectors, token_array = create_input_output_vectors(
        response_vectors, response_tokens, prerequisites)

# initiate 
model = Sequential()
# add layers, in the first layer
# units: dimensionality of the output space
units = 30
# input dimensions, the size of each embedding
input_dim = 30 
model.add(Dense(units = units, activation = 'sigmoid', input_dim = input_dim)) 

# sample random array for train
full_sample =  range(0,len(prereq_input_vectors))
train_sample =  random.sample(full_sample,1000)
print(len(train_sample))
test_sample = [x for x in full_sample if x not in train_sample]
print(len(test_sample))
x_train = prereq_input_vectors[train_sample]
y_train = prereq_output_vectors[train_sample]
x_test = prereq_input_vectors[test_sample]
y_test = prereq_output_vectors[test_sample]
match_token_test = [token_array[sample] for sample in test_sample]
token_target_test = [pair[0] for pair in match_token_test] 
token_prereq_test = [pair[1] for pair in match_token_test] 


model.compile(loss = 'mean_squared_error',
    optimizer = 'adam',
    metrics =['accuracy'])
model.fit(x_train, y_train, 
    epochs = 20, 
    batch_size = 10)


# [TODO] calculate the cosine similarity match for test
# And recalc accuracy based on how often a match is made
#score, acc = model.evaluate(x_test, y_test)
y_prediction = model.predict(x_test)

predicted_token_match = generate_similarity_tokens(method = 'cosine',
        target_vectors = y_prediction,
        target_tokens = token_target_test,
        comparison_vectors = response_vectors,
        comparison_tokens = response_tokens, 
        sample_number = 1)


def is_match_to_token(predicted_token_match, match_token_test):
    '''
        for each item in the test match see if the prediction matches
        match token
        the prediction match is in this format ('addition_0', ['addition_1|true'])
        the actual match is in this format ('addition_0', 'addition_1')
        see how often the exerices match each other 
    '''
    is_matches = []
    for i, predict in enumerate(predicted_token_match):
        true_match = match_token_test[i][1]
        predicted_match = predict[1][0].split('|')[0]
        if true_match == predicted_match:
            is_matches.append(True)
        else:
            is_matches.append(False)
    return is_matches

is_matches = is_match_to_token(predicted_token_match, match_token_test)
print(np.mean(is_matches))

