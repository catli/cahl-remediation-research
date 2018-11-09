from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.test.utils import common_texts, get_tmpfile, datapath
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

    def __init__(self, embedding_vectors, embedding_tokens):
        # vectors_file_name, tokens_file_name):
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

def main():
    root_path = os.path.split(os.getcwd())[0] + '/'
    print('root path: '+ root_path)
    print('read file: '+ read_file_affix)
    print('method: '+ method)
    print('sample_number: ' +str(remediation_sample_number))
    print('nearest comparison: ' + find_nearest_comparison)
    response_vectors = read_embedding_vectors(root_path +
                            'cahl_output/embed_vectors_' + read_file_affix)
    response_tokens = read_tokens(root_path +
                            'cahl_output/embed_index_' + read_file_affix)
    similarity_instance = CreateSimilarityToken(response_vectors, response_tokens)

    # find match between learning staste and response tokens
    if find_nearest_comparison == 'response':
        remediation_match_tokens = similarity_instance.find_similar_tokens(method = method,
                        sample_number = remediation_sample_number,
                        target_vectors = similarity_instance.learning_vectors,
                        target_tokens = similarity_instance.learning_state_tokens,
                        # change to compare against learning state tokens
                        comparison_vectors = similarity_instance.response_vectors,
                        comparison_tokens = similarity_instance.response_tokens)
    elif find_nearest_comparison == 'learn':
        remediation_match_tokens = similarity_instance.find_similar_tokens(method = method,
                        sample_number = remediation_sample_number,
                        target_vectors = similarity_instance.learning_vectors,
                        target_tokens = similarity_instance.learning_state_tokens,
                        # change to compare against learning state tokens
                        comparison_vectors = similarity_instance.learning_vectors,
                        comparison_tokens = similarity_instance.learning_state_tokens)

    # find match between learning staste and response tokens
    response_similar_tokens = similarity_instance.find_similar_tokens(method = method,
                    sample_number = 1,
                    target_vectors = similarity_instance.response_vectors,
                    target_tokens = similarity_instance.response_tokens,
                    comparison_vectors = similarity_instance.response_vectors,
                    comparison_tokens = similarity_instance.response_tokens)
    print('***CREATE RESPONSE TOKEN**')
    write_output(similarity = similarity_instance,
            root_path = root_path,
            remediation_match_tokens = remediation_match_tokens,
            response_similar_tokens = response_similar_tokens)



if __name__ == "__main__":
    read_file_affix = 'exercise'
    method = 'cosine'
    find_nearest_comparison = 'response' # response, learn (True-False)
    remediation_sample_number = 1
    main()


