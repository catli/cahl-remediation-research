from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
import os
# visualization 
import csv
import time
import pdb


from util_functions import find_most_similar_item_between_vectors
from util_functions import sample_highest_similarity_token_excl_self
from util_functions import read_embedding_vectors
from util_functions import read_tokens
from util_functions import write_token_file
from util_functions import write_vector_file
from util_functions import write_similarity_token_file


class CreateSimilarityToken:

    def __init__(self, embedding_vectors, embedding_tokens, learning_state_vectors, learning_state_tokens):
        # [TODO] add learning state tokens that are read in
        # vectors_file_name, tokens_file_name):
        self.response_vectors = embedding_vectors
        self.response_tokens  = embedding_tokens
        self.learning_state_tokens = learning_state_tokens
        self.learning_state_vectors = learning_state_vectors

    def generate_similarity_match(self, find_nearest_comparison, method,
            sample_number):
        # find match between learning staste and response tokens
        if find_nearest_comparison == 'response':
            match_tokens = generate_similarity_tokens(
                method = method,
                sample_number = sample_number,
                target_vectors = self.learning_state_vectors,
                target_tokens = self.learning_state_tokens,
                # change to compare against learning state tokens
                comparison_vectors = self.response_vectors,
                comparison_tokens = self.response_tokens)
        elif find_nearest_comparison == 'learn':
            match_tokens = generate_similarity_tokens(
                method = method,
                sample_number = sample_number,
                target_vectors = self.learning_state_vectors,
                target_tokens = self.learning_state_tokens,
                # change to compare against learning state tokens
                comparison_vectors = self.learning_state_vectors,
                comparison_tokens = self.learning_state_tokens)
        elif find_nearest_comparison == 'response-response':        
        # find match between learning staste and response tokens
            match_tokens = generate_similarity_tokens(
                method = method,
                sample_number = sample_number,
                target_vectors = self.response_vectors,
                target_tokens = self.response_tokens,
                comparison_vectors = self.response_vectors,
                comparison_tokens = self.response_tokens)
        return match_tokens    
        


def generate_similarity_tokens( method, target_vectors, target_tokens,
    comparison_vectors, comparison_tokens, sample_number):
    '''
    Input: Assume response vector and learning vector generated before
    similarity calculated
    Create generic function to find similar response token
    Using either euclidean distance or cosine distance function
    '''
    response_similarity_tokens = []
    similarity_loc = find_most_similar_item_between_vectors(method = method,
	    num_loc = 100, 
	    vectors_i = target_vectors,
	    vectors_j = comparison_vectors)
    for i, token in enumerate(target_tokens):
	# for each token, identify the location of similarity token
	vectors_most_similar_loc = similarity_loc[i]
	# the loc specifies the location of the highest similarity
	similar_token = sample_highest_similarity_token_excl_self(token,
	    comparison_tokens, vectors_most_similar_loc, method,
	    sample_number = sample_number)
	response_similarity_tokens.append((token, similar_token))
    return response_similarity_tokens



def test_create_similarity_token():
    # Check that the number of problem type token is same as original data set 
    # [TODO] update test so that learning state is an input
    test_vectors = np.array([[1,2,2,1],[1,1,0,1],[1,1,2,2.1],
        [-1,-3,1,2],[-1,-2,1,1],[0,1,0,-1],[1,-1,0,1]])
    test_tokens = [
        'addition_1|problem-type-0|true',
        'counting-out-1-20-objects|A|true',
        'addition_1|problem-type-0|false',
        'evaluating-expressions-in-two-variables-2|RationalNumbers|true',
        'evaluating-expressions-in-two-variables-2|RationalNumbers|false',
        'similar_to_addition_1|problem_type|true',
        'similar_to_evaluating_expressions|problem_type|true']
    test_similarity = CreateSimilarityToken(test_vectors, test_tokens)
    test_similar_tokens = test_similarity.find_similar_tokens(method = "cosine",
                    sample_number = 1,
                    target_vectors = test_similarity.response_vectors,
                    target_tokens = test_similarity.response_tokens,
                    comparison_vectors = test_similarity.response_vectors,
                    comparison_tokens = test_similarity.response_tokens)
    test_learning_tokens = test_similarity.find_similar_tokens(method = "cosine",
                    sample_number = 1,
                    target_vectors = test_similarity.learning_state_vectors,
                    target_tokens = test_similarity.learning_state_tokens,
                    comparison_vectors =test_similarity.response_vectors,
                    comparison_tokens = test_similarity.response_tokens)
    expected_learning_tokens = [
        ('evaluating-expressions-in-two-variables-2|RationalNumbers',
        'similar_to_evaluating_expressions|problem_type|true'),
        ('addition_1|problem-type-0', 'similar_to_addition_1|problem_type|true')]
    expected_similar_tokens = [('addition_1|problem-type-0|true',
            'addition_1|problem-type-0|false'),
        ('counting-out-1-20-objects|A|true',
            'addition_1|problem-type-0|false'),
        ('addition_1|problem-type-0|false',
            'addition_1|problem-type-0|true'),
        ('evaluating-expressions-in-two-variables-2|RationalNumbers|true',
            'evaluating-expressions-in-two-variables-2|RationalNumbers|false'),
        ('evaluating-expressions-in-two-variables-2|RationalNumbers|false',
            'evaluating-expressions-in-two-variables-2|RationalNumbers|true'),
        ('similar_to_addition_1|problem_type|true',
            'addition_1|problem-type-0|true'), 
        ('similar_to_evaluating_expressions|problem_type|true', 
            'evaluating-expressions-in-two-variables-2|RationalNumbers|true')]
    assert test_learning_tokens.sort() == expected_learning_tokens.sort()
    assert test_similar_tokens.sort() == expected_similar_tokens.sort()
    print('PASSES TEST!!')



def write_output( similarity,  root_path, path_affix, **kwargs):
    # write the output
    # [TODO] no need to write learning state tokens
    # [TODO] just write remediation matches 
    analysis_path = root_path + '/' +  'cahl_analysis' + '/' + path_affix + '/'
    # create directory if not already exist
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
    # this file should have the same number of state and be in the same order as above
    write_token_file(path = analysis_path, 
        file_name = 'learning_state_tokens',
        tokens = similarity.learning_state_tokens)
    for key, value in kwargs.items():
        write_similarity_token_file(path = analysis_path,
            file_name = key,
            similarity_tokens = value)

def create_path_affix(method, find_nearest_comparison, read_file_affix, remediation_sample_number):
    path_affix = method + '_' + find_nearest_comparison + '_' + read_file_affix + 'r' + str(remediation_sample_number)
    return path_affix


def create_file_path(path, file_name, file_affix):
    return path + file_name + file_affix

#########################################
# [TODO] create the average exercise embedding
# [TODO] run and create same output for exercise embedding

def create_similar_token(read_file_affix, method, find_nearest_comparison, remediation_sample_number):
    root_path = os.path.split(os.getcwd())[0] 
    print('root path: '+ root_path)
    print('read file: '+ read_file_affix)
    print('method: '+ method)
    print('sample_number: ' +str(remediation_sample_number))
    print('nearest comparison: ' + find_nearest_comparison)
    # [TODO] incorporate window and embedding into read_file_affix
    output_path = root_path + '/'  + 'cahl_output' + '/'
    response_vectors = read_embedding_vectors(create_file_path( output_path, 
                                    'embed_vectors_', read_file_affix))
    response_tokens = read_tokens(create_file_path( output_path,
                                    'embed_index_', read_file_affix))
    learning_state_vectors = read_embedding_vectors(create_file_path( output_path,                                                  'learning_state_vectors_', read_file_affix))
    learning_state_tokens = read_tokens(create_file_path( output_path,                                                               'learning_state_tokens_', read_file_affix))
    similarity_instance = CreateSimilarityToken(response_vectors, response_tokens, 
        learning_state_vectors, learning_state_tokens)

    # find match between learning staste and response tokens
    remediation_match_tokens = similarity_instance.generate_similarity_match(
                find_nearest_comparison = find_nearest_comparison,
                method = method,
                sample_number = remediation_sample_number)
    
    # find match between learning staste and response tokens
    response_similar_tokens = similarity_instance.generate_similarity_match(
                find_nearest_comparison = 'response-response',
                method = method,
                sample_number = 1)
    
    print('***CREATE RESPONSE TOKEN**')
    path_affix = create_path_affix(method, find_nearest_comparison, read_file_affix, remediation_sample_number)
  
    write_output(similarity = similarity_instance,
            root_path = root_path,
            path_affix = path_affix,
            remediation_match_tokens = remediation_match_tokens,
            response_similar_tokens = response_similar_tokens)



if __name__ == "__main__":
    # [TODO] append window and embedding to affix 
    window_size = 10
    embed_size = 30
    read_file_affix = 'full' + 'w' + str(window_size) + 'e' + str(embed_size)
    method = 'cosine'
    find_nearest_comparison = 'response' # response, learn (True-False)
    remediation_sample_number = 10
    create_similar_token(read_file_affix, method, find_nearest_comparison, remediation_sample_number)


