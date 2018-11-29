import numpy as np
import os
import csv
import pdb


##########################################
# Functions to find the most similar item between vectors

def find_most_similar_item_between_vectors(method, num_loc, vectors_i, vectors_j = None):
    '''
        calculate similarity between all possible vectors in set i and j
        using the method selected (cosine or euclidean)

        Input: matrix with vectors i and vectors j, they need to have the 
            same number of columns
        Output: with vectors_i as the target, find the location of the most similar
            vectors_i for each vector in vectors_j
    '''
    # if second set of vectors is none, assume finding euclidean distance
    # assume each vector compared against other vectors in vectors_i
    if vectors_j is None:
        vectors_j = vectors_i
    if method == "cosine":
        # create a map of min cosine distace where each row i and
        # column j represents the similarity between vector i
        # in the set of target vectors and vector j in the
        # comparison vectors set
        # similarity location find the location of
        # the (num_loc)th highest cosine similarity
        print('calculating cosine similarity')
        similarity_array = calculate_cosine_similarity(
                            vectors_i = vectors_i,
                            vectors_j = vectors_j)
        similarity_loc = find_the_max_loc(
                            similarity_array = similarity_array,
                            num_loc = num_loc)
    else:
        # create a map of min euclidean distace where each row i and
        # column j represents the similarity between vector i
        # in the set of target vectors and vector j in the
        # comparison vectors set
        # similarity location find the location of
        # the (num_loc)th small euclidean distance
        print('calculating euclidean similarity')
        similarity_array = calculate_euclidean_distance(
                            vectors_i = vectors_i,
                            vectors_j = vectors_j)
        similarity_loc = find_the_min_loc(
                            similarity_array = similarity_array,
                            num_loc = num_loc)
    return similarity_loc


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
    vector_dot_product = np.dot(vectors_i, np.array(vectors_j).T)
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
    similarity_item_num  = len(similarity_array[0])
    if num_loc > similarity_item_num:
        # if the num of similar item retrieved is greater than the available item
        # than set to num of similar item
        num_loc = similarity_item_num
    least = np.argpartition(similarity_array, num_loc)[:, :num_loc]
    return least

def sample_highest_similarity_token_excl_self(target_token, response_tokens, similarity_locs,
        method, sample_number):
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
    filtered_similar_tokens = list(filter(lambda x: target_exercise!=x.split("|")[0], similar_tokens))
    # if more than one similar token selected
    # then select the last one in the sorted list
    # sample 5, 10, 20 (for cosine)
    if len(filtered_similar_tokens)>1 and method =='cosine':
        filtered_similar_tokens = filtered_similar_tokens[-sample_number: ]
    elif len(filtered_similar_tokens)>1:
        filtered_similar_tokens = filtered_similar_tokens[:sample_number]
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
