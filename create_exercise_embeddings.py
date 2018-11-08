
import os
import numpy as np
import pdb


from util_functions import read_embedding_vectors
from util_functions import read_tokens
from util_functions import write_token_file
from util_functions import write_vector_file


##########################################
# Convert average the response-level embedding
# into exercise-level embedding

def group_token_and_vectors_by_exercise(response_tokens, response_vectors):
    '''
        create a dicitonary of tokens and vectors, grouped by exercise
        input: array of response tokens and associated response vectors
        output: a dictionary of grouped embedding
    '''
    grouped_embeddings = {}
    for i, token in enumerate(response_tokens):
        token_split = token.split('|')
        exercise_token = token_split[0] + '|' + token_split[2]
        response_vector = response_vectors[i]
        if exercise_token in grouped_embeddings:
            vectors = grouped_embeddings[exercise_token]['vectors']
            grouped_embeddings[exercise_token]['tokens'].append(token)
            vectors = np.insert(vectors, len(vectors), response_vector, axis = 0)
            grouped_embeddings[exercise_token]['vectors'] = vectors
        else:
            grouped_embeddings[exercise_token] = {}
            grouped_embeddings[exercise_token]['vectors'] = np.array([response_vector])
            grouped_embeddings[exercise_token]['tokens'] = [token]
    return grouped_embeddings


def average_vectors_by_exercise(grouped_embeddings):
    '''
        average the vectors of the each grouped exercise
        input: grouped embedding ditionary
        output: token array of exericse token and the average vector for
            each token
    '''
    exercise_tokens = []
    exercise_vectors = []
    for item in grouped_embeddings:
        exercise_tokens.append(item)
        avg_exercise_vector = np.mean(grouped_embeddings[item]['vectors'], axis=0)
        exercise_vectors.append(avg_exercise_vector)
    return exercise_tokens, exercise_vectors



def main():
    root_path = os.path.split(os.getcwd())[0] + '/'
    output_path = root_path + 'cahl_output' + '/'
    read_file_affix = 'full'

    response_vectors = read_embedding_vectors(root_path +
                            'cahl_output/embed_vectors_' +
                            read_file_affix)
    response_tokens = read_tokens(root_path +
                            'cahl_output/embed_index_' +
                            read_file_affix)
    grouped_embeddings = group_token_and_vectors_by_exercise(response_tokens, response_vectors)
    exercise_tokens, exercise_vectors = average_vectors_by_exercise(grouped_embeddings)

    write_token_file( path = output_path,
               file_name = 'embedding_index_exercise',
               tokens = exercise_tokens)
    write_vector_file( path = output_path,
               file_name = 'embedding_vectors_exercise',
               vectors = exercise_vectors)


if __name__ == "__main__":
    main()


