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

##########################################
# Functions to calculate cosine similarity 
def calculate_cosine_similarity(vectors_i, vectors_j = None):
    '''
        create the cosine similarity matrix
        when the target embedding vector is pulled from a different
        set of vectors as the comparison set 
        used for when the target vector represent the comparison set 
        [TODO] Calculate a similar approach using eculidean distance 
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
# Functions find the highest similarity  
def find_the_max_loc(similarity_array, num):
    '''
        For each row of the data array, find the location of the top {num}
        values and return a list of the location
    '''
    top = np.argpartition(similarity_array, -num)[:, -num:]
    return top
    
def create_highest_similarity_token_excl_self(target_token, response_tokens, similarity_locs, loc):
    '''
        Input: target token, entire set of response tokens and location highest similarity
            response tokens 
        Output: the most similar token to the target token 
        Exclude any similar token that are exact matches of target
    '''
    similar_tokens = [response_tokens[loc] for loc in similarity_locs]
    # exclude similar tokens that are the same as target 
    # and pick the last token
    similar_token = list(filter(lambda x: target_token  not in x, similar_tokens))
    if len(similar_token)>1:
        similar_token = similar_token[loc]
    return ''.join(similar_token)


##########################################
# Functions to write files  

def write_token_file(file_name, tokens):
    '''write individual tokens into flat file. Estimated time: 1.5 sec for 1M rows'''
    path = os.path.expanduser('~/analysis/'+file_name+'.csv')
    print(path)
    open_file = open(path, "w")
    with open_file:
        [open_file.write(token + '\n') for token in tokens]
        
def write_vector_file(file_name, vectors):
    path = os.path.expanduser('analysis/'+file_name+'.out')
    print(path)
    np.savetxt(path, vectors, delimiter = ',')      
        
def write_similarity_token_file(file_name, similarity_tokens):
    '''write tokens into flat file, can handle a tuple of tokens per line for similarity tokens'''
    path = os.path.expanduser('~/analysis/'+file_name+'.csv')
    print(path)
    open_file = open(path, "w")
    with open_file:
        csvwriter = csv.writer(open_file, delimiter = ',')
        csvwriter.writerows(similarity_tokens)
 

class CreateSimilarityToken:
    
    def __init__(self, vectors_file_name, tokens_file_name):
        self.read_embedding_vectors(vectors_file_name)
        self.read_tokens(tokens_file_name)
       

    def read_embedding_vectors(self, file_name):
        '''
            read in the response embedding vector files
            these were stored using np.savetxt (.out)
            return a matrix of embedding vectors
        '''
        path = os.path.expanduser('~/output/'+file_name+'.out')
        self.response_vectors = np.loadtxt(path, delimiter = ',')

    def read_tokens(self, file_name):
        '''
            read in the flat file containing the list of response tokens
            return an array of response tokens associated with embedding vectors
        '''
        path = os.path.expanduser('~/output/'+file_name+'.csv')
        reader = open(path,'r')
        self.response_tokens = []
        for line in reader:
            self.response_tokens.append(line.strip())

    def create_similar_learning_token_from_response_token(self):
        '''
            aggregates learning state tokens 
        '''
        self.create_unique_problem_type_token()
        self.create_learning_state_embedding()
        self.find_similar_learning_token()
        
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


    def find_similar_learning_token(self):
        '''
            Input: The learning tokens, learning tokens, response embedding and response tokens
            Output: A list of response response token that most closely approximates each learning state
                (true vector - false vector)
        '''
        cos_similarity_array = calculate_cosine_similarity(self.learning_vectors, 
                                                                            self.response_vectors)
        # sorting in advance helps to reduce to reduce the runtime by 50% 
        # can define number of top item in the parameter 
        max_cos_similarity_loc = find_the_max_loc(cos_similarity_array, 3)
        self.learning_similarity_tokens = []
        for i, token in enumerate(self.learning_state_tokens):
            # for each learning state token, find the most similar response token 
            # for a different problem 
            # [TODO] for euclidean distance find min rather than max with sort
            print(token)
            print(self.response_tokens)
            print(max_cos_similarity_loc[i])
            vectors_most_similar_loc = max_cos_similarity_loc[i]
            # the loc specifies the location of the highest similarity
            # in this case -1 is the last item in the ordered array
            similar_token = create_highest_similarity_token_excl_self(token, 
                                self.response_tokens, vectors_most_similar_loc, -1)
            self.learning_similarity_tokens.append((token, similar_token))
        print("*highest similarity tokens to learning tokens create*")

        
    def find_similar_response_token(self):
        '''
            find the most similar token based on cosine similarity matrix 
            input: vectors - the embedding vectors for each token 
            index: token names or the name associated with the average embeddings 
        '''
        cos_similarity_array = calculate_cosine_similarity(self.response_vectors)
        # the top 2 most similar tokens (includes self)
        max_cos_similarity_loc = find_the_max_loc(cos_similarity_array, 2)
        self.response_similarity_tokens = []
        for i, token in enumerate(self.response_tokens):
            # for each token, find the second largest similarity
            # the highest similarity is almost always itself
            vectors_most_similar_loc = max_cos_similarity_loc[i]
            # the loc specifies the location of the highest similarity
            # in this case since self is excluded, the first item of sorted array should
            # be the only item and would be returned with 0
            similar_token = create_highest_similarity_token_excl_self(token, 
                                self.response_tokens, vectors_most_similar_loc, 0)
            self.response_similarity_tokens.append((token, similar_token))
        print("*highest similarity tokens to learning tokens create*")
 


def tests_create_similarity_token():
    # Check that the number of problem type token is same as original data set 
    learning_similarity = CreateSimilarityToken('embed_vectors_test','embed_index_test')
    learning_similarity.create_similar_learning_token_from_response_token()
    learning_similarity.find_similar_response_token()

    
# [TODO] create assertion test output 
# Expected output for similar learning token
# [('evaluating-expressions-in-two-variables-2|RationalNumbers', 'similar_to_evaluating_expressions|problem_type|true'), ('addition_1|problem-type-0', 'similar_to_addition_1|problem_type|true')]
# Expected output for similar token output
# [('addition_1|problem-type-0|true', 'addition_1|problem-type-0|false'), ('counting-out-1-20-objects|A|true', 'addition_1|problem-type-0|false'), ('addition_1|problem-type-0|false', 'addition_1|problem-type-0|true'), ('evaluating-expressions-in-two-variables-2|RationalNumbers|true', 'evaluating-expressions-in-two-variables-2|RationalNumbers|false'), ('evaluating-expressions-in-two-variables-2|RationalNumbers|false', 'evaluating-expressions-in-two-variables-2|RationalNumbers|true'), ('similar_to_addition_1|problem_type|true', 'addition_1|problem-type-0|true'), ('similar_to_evaluating_expressions|problem_type|true', 'evaluating-expressions-in-two-variables-2|RationalNumbers|true')]

# OTHER ASSERTIONS
# Check that the total number of tokens generated matches input 


#########################################
learning_similarity = CreateSimilarityToken('embed_vectors_full','embed_index_full')
learning_similarity.create_similar_learning_token_from_response_token()
learning_similarity.find_similar_response_token()
write_similarity_token_file('learning_similar_tokens'
                            ,learning_similarity.learning_similarity_tokens)
# this file should have the same number of state and be in the same order as above
write_token_file('learning_state_tokens'
                            ,learning_similarity.learning_state_tokens)
write_vector_file('problemtype_false_vectors'
                            ,learning_similarity.false_vectors)
write_vector_file('problemtype_true_vectors'
                            ,learning_similarity.true_vectors)
write_vector_file('learning_vectors'
                            ,learning_similarity.learning_vectors)


