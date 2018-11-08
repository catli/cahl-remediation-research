from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.test.utils import common_texts, get_tmpfile, datapath
import pandas as pd
import numpy as np
import os
# visualization 
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import csv
import time
import pdb

##########################################
# Functions to calculate cosine similarity 
def calculate_cosine_similarity(vectors_i, vectors_j = None):
    '''
        create the cosine similarity matrix between each possible vectors set
        i and j. If vectors_j is None, then assume vectors_i 
        compared against vectors in the same set. When vectors_j is None,
        then assume vector set i compared against a different vector set j
        Output: vector representing cosine similarity of size I x J
            each row i and column has value =  v_i (v_j.T) / ||v_i|| * ||v_j|| 
    '''
    # if second set of vectors is none, assume finding cosine similarity
    # assume each vector compared against other vectors in vectors_i
    if vectors_j is None:
        vectors_j = vectors_i
    vector_dot_product = np.dot(vectors_i, vectors_j.T)
    inv_norm = find_inv_norm_product(vectors_i, vectors_j)
    # use hadamard multiplication to calculate multiplication
    return np.multiply(vector_dot_product, inv_norm)

def find_inv_norm_product(vectors_i, vectors_j = None):
    '''
        input: given two sets of vectors
        output: the product of norms for each possible i and j combination
            each row i and column j has value =  ||v_i|| * ||v_j|| 
    '''
    inv_norm_i = find_inv_norm_of_matrix(vectors_i)
    inv_norm_j = find_inv_norm_of_matrix(vectors_j)
    return np.dot(inv_norm_i[:,None], inv_norm_j[None,:])

def find_inv_norm_of_matrix( vectors):
    '''
        intput: array of vectors
        output: norm for a set of vectors
    '''
    vec_norm = [np.linalg.norm(vector) for vector in vectors]
    return 1/np.array(vec_norm)


##########################################
# Functions to calculate euclidean distance
def calculate_euclidean_distance(vectors_i, vectors_j = None):
    '''
        create the euclidean distance between each possible vectors set
        i and j. If vectors_j is None, then assume vectors_i
        compared against vectors in the same set. When vectors_j is None,
        then assume vector set i compared against a different vector set j.
        vector set i need to have the same number of oclumns as vector set j
        Output: vector representing euclidean distance of size I x J
    '''
    # if second set of vectors is none, assume finding euclidean distance
    # assume each vector compared against other vectors in vectors_i
    if vectors_j is None:
        vectors_j = vectors_i
    # subtract every vectors_i from every possible vectors_j
    euclidean_norms = find_euclidean_norm(vectors_i, vectors_j)
    # use hadamard multiplication to calculate multiplication
    return euclidean_norms

def find_euclidean_norm(vectors_i, vectors_j):
    '''
        input: given two sets of vectors, vectors i and vectors j
            with same number of columns. where i has dimension
            I x E and j has dimension J x E
        output: a matrix with the euclidean norm comparing every vector in
            vectors i against every vector in vectors j, with dimension IxJ
    '''
    euclidean_norms =[]
    for vector_i in vectors_i:
        vec_norm = find_vector_euclidean_norm(vector_i, vectors_j)
        euclidean_norms.append(vec_norm)
    return euclidean_norms

def find_vector_euclidean_norm( vector_i, vectors_j):
    '''
        input: a single vector_i and a set of vectors_j, with same number
            of columns
        output: the square distance 
            for each column j value = || v_i - v_j || 
    '''
    diff_vectors = vector_i - vectors_j
    vec_norm = [np.linalg.norm(vector) for vector in diff_vectors]
    return vec_norm

##########################################
# Functions find the highest similarity
def find_the_max_loc(similarity_array, num_loc):
    '''
        For each row of the data array, find the location of the top {num}
        values and return a list of the location
    '''
    similarity_item_num  = similarity_array.shape[1]
    if num_loc > similarity_item_num:
        # if the num of similar item retrieved is greater than the available item
        # than set to num of similar item
        num_loc = similarity_item_num
    top = np.argpartition(similarity_array, -num_loc)[:, -num_loc:]
    return top

def find_the_min_loc(similarity_array, num_loc):
    '''
        For each row of the data array, find the location of the top {num}
        values and return a list of the location
    '''
    least = np.argpartition(similarity_array, num_loc)[:, :num_loc]
    return least

def sample_highest_similarity_token_excl_self(target_token, response_tokens, similarity_locs, sample_number):
    '''
        Input: target token, entire set of response tokens and location highest similarity
            response tokens
        Output: the most similar token to the target token
        Exclude any similar token that are exact matches of target
            and any tokens where the exercise matches
    '''
    similar_tokens = [response_tokens[similarity_loc] for similarity_loc in similarity_locs]
    # find the target exercise (remove the problem type)
    target_exercise = target_token.split("|")[0]
    # exclude similar tokens that below to the same exercise
    # as the target
    # [TODO]: make this flexible so the exclusion
    filtered_similar_tokens = filter(lambda x: target_exercise!=x.split("|")[0],
                        similar_tokens)
    # if more than one similar token selected
    # then select the last one in the sorted list
    # sample 5, 10, 20 
    if len(filtered_similar_tokens)>1:
        filtered_similar_tokens = filtered_similar_tokens[-sample_number: ]
    return filtered_similar_tokens


##########################################
# Functions to read and write files

def read_embedding_vectors(file_name):
    '''
        read in the response embedding vector files
        these were stored using np.savetxt (.out)
        return a matrix of embedding vectors
        # TESTING
    '''
    path = os.path.expanduser(file_name+'.out')
    response_vectors = np.loadtxt(path, delimiter = ',')
    return response_vectors

def read_tokens(file_name):
    '''
        read in the flat file containing the list of response tokens
        return an array of response tokens associated with embedding vectors
    '''
    path = os.path.expanduser(file_name+'.csv')
    reader = open(path,'r')
    response_tokens = []
    for line in reader:
        # drop any quotes in the token
        clean_line = line.replace("'","").replace('"','')
        response_tokens.append(clean_line.replace("'","").strip())
    return response_tokens

def write_token_file(path, file_name, tokens):
    '''write individual tokens into flat file. Estimated time: 1.5 sec for 1M rows'''
    path = os.path.expanduser(path+file_name+'.csv')
    print(path)
    open_file = open(path, "w")
    with open_file:
        [open_file.write(token + '\n') for token in tokens]

def write_vector_file(path, file_name, vectors):
    path = os.path.expanduser(path+file_name+'.out')
    print(path)
    np.savetxt(path, vectors, delimiter = ',')

def write_similarity_token_file(path, file_name, similarity_tokens):
    '''write tokens into flat file, can handle a tuple of tokens per line for similarity tokens'''
    path = os.path.expanduser(path+file_name+'.tsv')
    print(path)
    open_file = open(path, "w")
    with open_file:
        # [ open_file.write(token[0] + '\t' + token[1])
        #     for token in similarity_tokens]
        csvwriter = csv.writer(open_file, delimiter = '\t')
        csvwriter.writerows(similarity_tokens)


class CreateSimilarityToken:

    def __init__(self, embedding_vectors, embedding_tokens):
        # vectors_file_name, tokens_file_name):
        self.response_vectors = embedding_vectors
        self.response_tokens  = embedding_tokens
        self.create_unique_problem_type_token()
        self.create_learning_state_embedding()


    def create_unique_problem_type_token(self):
        '''
            Create a list of unique problem type tokens
            Input:  a list of all exercise - problemtype - is response correct
                token, i.e fractions|problemtype1|true or fractions|problemtype1|false
            Output: an array of of unique exercise - problem type tokens,
                i.e. the above would be combined into one token fractions|problemtype1
        '''
        undup_problem_type_tokens = []
        for token_array in self.response_tokens:
             new_token = "|".join(token_array.split('|')[0:-1])
             undup_problem_type_tokens.append(new_token)
        self.problem_type_tokens = list(set(undup_problem_type_tokens))

    def create_learning_state_embedding(self):
        '''
            Input: problem type tokens
            Output: embedding vectors representing learning state (true vector - false vector)
            for each problem type
        '''
        self.learning_vectors = []
        self.true_vectors = []
        self.false_vectors = []
        self.learning_state_tokens = []
        self.missing_learning_tokens = []
        for token in self.problem_type_tokens:
            try:
                true_token = token + '|' + 'true'
                false_token = token + '|' + 'false'
                # find true and false location
                true_vector = self.response_vectors[ self.response_tokens.index(true_token)] 
                false_vector = self.response_vectors[ self.response_tokens.index(false_token)]
                self.true_vectors.append(true_vector)
                self.false_vectors.append(false_vector)
                self.learning_vectors.append(true_vector - false_vector)
                self.learning_state_tokens.append(token)
            except ValueError:
                # if missing token
                self.missing_learning_tokens.append(token)


    def find_similar_tokens(self, method, target_vectors, target_tokens,
                comparison_vectors, comparison_tokens, sample_number=1):
        '''
            Input: The learning tokens, learning tokens, response embedding and response tokens
            Output: A list of response response token that most closely approximates each learning state
                (true vector - false vector)
        '''
        similarity_tokens = self.generate_similarity_tokens(
                                method = method,
                                target_vectors = target_vectors,
                                target_tokens = target_tokens,
                                comparison_vectors = comparison_vectors,
                                comparison_tokens = comparison_tokens,
                                sample_number = sample_number)
        print("*similar tokens matched*")
        return similarity_tokens



    def generate_similarity_tokens(self, method, target_vectors, target_tokens,
            comparison_vectors, comparison_tokens, sample_number):
        '''
        Input: Assume response vector and learning vector generated before
            similarity calculated
        Create generic function to find similar response token
        Using either euclidean distance or cosine distance function
        '''
        response_similarity_tokens = []
        if method == "cosine":
            # create a map of min cosine distace where each row i and
            # column j represents the similarity between vector i
            # in the set of target vectors and vector j in the
            # comparison vectors set
            # similarity location find the location of
            # the (num_loc)th highest cosine similarity
            similarity_array = calculate_cosine_similarity(
                                vectors_i = target_vectors,
                                vectors_j = comparison_vectors)
            similarity_loc = find_the_max_loc(
                                similarity_array = similarity_array,
                                num_loc = 100)
        else:
            # create a map of min euclidean distace where each row i and
            # column j represents the similarity between vector i 
            # in the set of target vectors and vector j in the 
            # comparison vectors set
            # similarity location find the location of 
            # the (num_loc)th small euclidean distance
            similarity_array = calculate_euclidean_distance(
                                vectors_i = target_vectors,
                                vectors_j = comparison_vectors)
            similarity_loc = find_the_min_loc(
                                similarity_array = similarity_array,
                                num_loc = 100)
        for i, token in enumerate(target_tokens):
            # for each token, identify the location of similarity token
            vectors_most_similar_loc = similarity_loc[i]
            # the loc specifies the location of the highest similarity
            if method == "cosine":
                similar_token = sample_highest_similarity_token_excl_self(token,
                                comparison_tokens, vectors_most_similar_loc,
                                sample_number = sample_number)
            else:
                similar_token = sample_highest_similarity_token_excl_self(token,
                                comparison_tokens, vectors_most_similar_loc, 
                                sample_number = sample_number)
            response_similarity_tokens.append((token, similar_token))
        return response_similarity_tokens



def test_create_similarity_token():
    # Check that the number of problem type token is same as original data set 
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
                    target_vectors = test_similarity.learning_vectors,
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



def write_output( similarity,  root_path, **kwargs):
    # write the output
    analysis_path = root_path + 'cahl_analysis/'
    # this file should have the same number of state and be in the same order as above
    write_token_file(path = analysis_path, 
           file_name = 'learning_state_tokens',
           tokens = similarity.learning_state_tokens)
    write_vector_file(path = analysis_path,
            file_name = 'response_false_vectors',
            vectors = similarity.false_vectors)
    write_vector_file(path = analysis_path,
            file_name = 'response_true_vectors',
            vectors = similarity.true_vectors)
    write_vector_file(path = analysis_path,
            file_name = 'learning_state_vectors',
             vectors = similarity.learning_vectors)
    for key, value in kwargs.items():
        write_similarity_token_file(path = analysis_path,
           file_name = key,
           similarity_tokens = value)


#########################################
# [TODO] create the average exercise embedding
# [TODO] run and create same output for exercise embedding

# RUN TEST WHEN RUNNING MODEL
# tests_create_similarity_token()

def main():
    root_path = os.path.split(os.getcwd())[0] + '/'
    print('root path: '+ root_path)
    response_vectors = read_embedding_vectors(root_path +
                            'cahl_output/embed_vectors_' + read_file_affix)
    response_tokens = read_tokens(root_path +
                            'cahl_output/embed_index_' + read_file_affix)
    similarity_instance = CreateSimilarityToken(response_vectors, response_tokens)

    # find match between learning staste and response tokens
    remediation_match_tokens = similarity_instance.find_similar_tokens(method = method,
                    sample_number = remediation_sample_number,
                    target_vectors = similarity_instance.learning_vectors,
                    target_tokens = similarity_instance.learning_state_tokens,
                    comparison_vectors =similarity_instance.response_vectors,
                    comparison_tokens = similarity_instance.response_tokens)

    # find match between learning staste and response tokens
    response_similar_tokens = similarity_instance.find_similar_tokens(method = method,
                    sample_number = 1,
                    target_vectors = similarity_instance.response_vectors,
                    target_tokens = similarity_instance.response_tokens,
                    comparison_vectors =similarity_instance.response_vectors,
                    comparison_tokens = similarity_instance.response_tokens)
    print('***CREATE RESPONSE TOKEN**')
    write_output(similarity = similarity_instance,
            root_path = root_path,
            remediation_match_tokens = remediation_match_tokens,
            response_similar_tokens = response_similar_tokens)


if __name__ == "__main__":
    read_file_affix = 'full'
    method = 'cosine'
    remediation_sample_number = 5
    main()


