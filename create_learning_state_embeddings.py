import numpy as np
import os
import csv
import time
import pdb


from util_functions import read_embedding_vectors
from util_functions import read_tokens
from util_functions import write_token_file
from util_functions import write_vector_file


class CreateSimilarityToken:

    def __init__(self, embedding_vectors, embedding_tokens):
        self.response_vectors = embedding_vectors
        self.response_tokens  = embedding_tokens
        self.create_unique_problem_type_token()
        self.create_learning_state_embedding()


    def create_unique_problem_type_token(self):
        # TODO: generalize for exercise level tokens
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
        self.learning_state_vectors = []
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
                self.learning_state_vectors.append(true_vector - false_vector)
                self.learning_state_tokens.append(token)
            except ValueError:
                # if missing token
                self.missing_learning_tokens.append(token)



def write_learning_state_output(similarity, root_path, read_file_affix):
    '''
       Input: the instance  
    '''
    # write the output
    output_path = root_path + '/' + 'cahl_output' + '/' 
    # this file should have the same number of state and be in the same order as above
    write_token_file(path = output_path, 
           file_name = create_file_name('learning_state_tokens_',read_file_affix),
           tokens = similarity.learning_state_tokens)
    write_vector_file(path = output_path,
           file_name = create_file_name( 'learning_state_vectors_',read_file_affix),
           vectors  = similarity.learning_state_vectors)
    

def create_file_name(file_type, read_file_affix):
    file_name = file_type +  read_file_affix   
    return file_name


#########################################

def create_learning_embedding(read_file_affix):
    '''
       Has to run in the folder where the file lives in order for the right directory
       to be called
    '''
    path = os.path.split(os.getcwd())
    root_path = path[0]  
    print('root path: '+ root_path)
    print('read file: '+ read_file_affix)
    response_vectors = read_embedding_vectors(root_path + '/' +
                            'cahl_output/embed_vectors_' + read_file_affix)
    response_tokens = read_tokens(root_path + '/' +
                            'cahl_output/embed_index_' + read_file_affix)
    similarity_instance = CreateSimilarityToken(response_vectors, response_tokens)

    # find match between learning staste and response tokens
    print('***CREATE LEARNING STATE TOKENS**')
    write_learning_state_output(similarity = similarity_instance,
            root_path = root_path,
            read_file_affix = read_file_affix)



if __name__ == "__main__":
    window_size = 10
    embed_size = 30
    read_file_affix = 'full' + 'w' + str(window_size) + 'e' + str(embed_size)  
    create_learning_embedding(read_file_affix)


