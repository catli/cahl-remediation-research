############################################
# [2] Visualize using t-NSE, a tool to visualize high dimensional data
# by converting to join probabilities and minimizing divergence
# we minimize it to an embedding with 2 dimensions  (n_components = 2)
import numpy as np
import pandas as pd
import os
import pdb
import ast

from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

from util_functions import read_embedding_vectors
from util_functions import read_tokens


##########################################
# Functions to read and write files
# [TODO] 
# [x] READ response embedding vector and embedding token
# [x] READ learning state vector and learning state token # 
# [x ] CREATE index for response and learning state and concatenate the vectors 
# [x ] RUN TSNE on concatenated vectors
# [x] Separate out the concatenated TSNE vectors into response and learning state
# [x ] READ remediation match token
# Create two set of learning state
# (1) response | remediation item (join response to itself)
# (2) response | learning state

# Create response true / false fields 
# Join topic tree 



class CreateTSNE:

    def __init__(self, response_vectors, learning_state_vectors, response_tokens, learning_state_tokens):
        '''
        input: response vectors and learning state (True - False) vectors 
        output: when initiated, create a 
        '''
        self.response_vectors = response_vectors
        self.learning_state_vectors = learning_state_vectors
        self.response_tokens  = response_tokens
        self.learning_state_tokens = learning_state_tokens
        self.concatenate_response_and_learning_state()
        self.create_tsne_of_vectors()
        self.create_dataframe_of_response_tsne()
        self.create_dataframe_of_learning_state_tsne()


    def concatenate_response_and_learning_state(self):
        '''
        Concatenate the response and learning state vectors
        along with the index for response vectors
        '''
        response_vector_size = len(self.response_vectors)
        learning_vector_size = len(self.learning_state_vectors)
        self.response_index = [0, response_vector_size]
        self.learning_index = [response_vector_size, 
                response_vector_size + learning_vector_size]
        self.concatenated_vectors = np.concatenate((self.response_vectors,
            self.learning_state_vectors), axis=0)

    def create_tsne_of_vectors(self):
        '''
        Generate the TSNE for the concatenated vectors 
        and output the repponse and learning tsne
        '''
        tsne_vectors = TSNE(n_components=2).fit_transform(self.concatenated_vectors)
        self.response_tsne = tsne_vectors[self.response_index[0]:self.response_index[1]] 
        self.learning_tsne = tsne_vectors[self.learning_index[0]:self.learning_index[1]]
     

    # [TODO] update functions so that they are creating the joined TSNE files 
    def create_dataframe_of_response_tsne(self):
        '''
            create a pandas dataset with the x, y coodinates (response tsne),
            the token name, exercise name , problem type (if available),
            and correctness of answer if available
        '''
        self.response_tsne_df = pd.DataFrame(columns = ['x','y','exercise',
            'problem_type','response'])
        for i, token in enumerate(self.response_tokens):
            vector = self.response_tsne[i]
            exercise = token.split('|')[0]
            problem_type = token.split('|')[0] + '|' + token.split('|')[1]
            response_accuracy = token.split('|')[2]
            self.response_tsne_df = self.response_tsne_df.append({
                                    'x': vector[0], 
                                    'y': vector[1],
                                    'exercise': exercise,
                                    'problem_type': problem_type,
                                    'response': response_accuracy
                                     }, ignore_index = True)

    def create_dataframe_of_learning_state_tsne(self):
        '''
            create a pandas dataset with the x, y coodinates (learning state tsne),
            the token name, exercise name , problem type (if available),
        '''
        self.learning_state_tsne_df = pd.DataFrame(columns = ['x2','y2',
            'exercise','problem_type'])
        for i, token in enumerate(self.learning_state_tokens):
            vector = self.learning_tsne[i]
            exercise = token.split('|')[0]
            problem_type = token.split('|')[0] + '|' + token.split('|')[1]
            self.learning_state_tsne_df = self.learning_state_tsne_df.append({
                                    'x2': vector[0], 
                                    'y2': vector[1],
                                    'exercise': exercise,
                                    'problem_type': problem_type
                                    }, ignore_index = True)


    def join_response_and_learning_state_df(self):
        '''
           join learning state data frame to response data frame 
        '''
        return pd.merge(self.response_tsne_df, 
                self.learning_state_tsne_df,
            left_on=['exercise', 'problem_type'], 
            right_on=['exercise', 'problem_type'], how='inner')


    def join_response_and_remediation_df(self, remediation_df):
        '''
           join learning state data frame to response data frame 
        '''
        joined_response_to_remediation_token = pd.merge(
                self.response_tsne_df, 
                remediation_df,
                left_on=['exercise', 'problem_type'], 
                right_on=['target_exercise', 'target_problem_type'], how='inner')
        response_copy = self.response_tsne_df.copy() 
        response_copy.rename(index =str, columns ={'x':'x2','y':'y2'})
        joined_response_to_remediation_tsne = pd.merge(
                joined_response_to_remediation_token,
                response_copy,  
                left_on=['predicted_exercise','predicted_problem_type', 'predicted_response'], 
                right_on=['exercise','problem_type','response'])
        return joined_response_to_remediation_tsne



def create_file_path(path, file_name, file_affix):
    return path + file_name + file_affix



def read_remediation_match(file_name):
    '''
        read in the flat file containing the list of response tokens
        return a data frame of response tokens associated with embedding vectors
    '''
    path = os.path.expanduser(file_name+'.tsv')
    reader = open(path,'r')
    remediation_df = pd.DataFrame(columns = ['target_exercise','target_problem_type',
        'predicted_exercise','predicted_problem_type', 'predicted_response','is_correct_match'])
    for line in reader:
        # delimit by comma
        splitline = line.strip().split('\t')
        target = splitline[0]
        # evaluate as array list of tokens
        remediation = ast.literal_eval(splitline[1])[0]
        is_correct_match = splitline[2]
        remediation_df = remediation_df.append({
            'target_exercise': target.split('|')[0],
            'target_problem_type': target.split('|')[0]+'|'+ target.split('|')[1],
            'predicted_exercise': remediation.split('|')[0],
            'predicted_problem_type': remediation.split('|')[0]+'|'+remediation.split('|')[1],
            'predicted_response':  remediation.split('|')[2],
            'is_correct_match': is_correct_match
            }, ignore_index = True)
    return remediation_df


def read_topic_tree_file(topic_file_name, topic_file_path):
    '''
        read in the topic tree and filter for exercises of one subject
        that way we can focus on patterns for one subject
    '''
    learning_similarity_exercises = {}
    for line in reader:
        splitline = line.strip().split(",")
        # split the text by "|" to eliminate problem type, keeping only
        # exercise
        exercise_key = splitline[0].split('|')[0]
        similar_exercise = splitline[1].split('|')[0]
        try:
            if exercise_key not in learning_similarity_exercises:
                learning_similarity_exercises[exercise_key].append(similar_exercise)
        except KeyError:
            learning_similarity_exercises[exercise_key] = similar_exercise
    return learning_similarity_exercises


def write_to_tsv(dataframe, file_name):
    '''
        write data frame to a tab delimited file
    '''
    file_name = file_name + '.tsv'
    dataframe.to_csv(file_name,
        sep = '\t', 
        header = True,
        index = False)




##########################################
# Functions to create and join data frame

def join_tsne_to_topic_tree(selected_subjects, tsne_df, tsne_join_column):
    '''
       join the tsne dataframe to the selected subjects
       by exercise name, 
    '''
    math_topic_tree_df = read_topic_tree_data_frame(file_name = 'math_topic_tree')
    subject_topic_tree_df = create_dataframe_of_topic_tree(
                            subjects = selected_subjects,
                            math_topic_tree = math_topic_tree_df)
    joined_topic_tsne = join_tsne_to_topic_tree_dataframe(
                            tsne_df = tsne_df, 
                            topic_tree_df = subject_topic_tree_df,
                            tsne_join_column = tsne_join_column)
    return joined_topic_tsne 


def join_tsne_to_topic_tree_dataframe(tsne_df, topic_tree_df, tsne_join_column):
    '''
       join the tsne dataframe tot he subject and topic tree
       shoudl be a many to many join
    '''
    return pd.merge(tsne_df, topic_tree_df, 
        left_on = tsne_join_column, right_on='exercise', how='inner')


def read_topic_tree_data_frame(file_name):
    topic_tree_df = pd.read_csv(file_name+'.csv')
    return topic_tree_df.drop(columns = ['id'])


def create_dataframe_of_topic_tree(subjects, math_topic_tree):
    '''
        create a pandas dataset with
        exercise, subject, unit, lesson
        for every subject that included in subjects
    '''
    subject_topic_tree = pd.DataFrame()
    for subject in subjects:
        subject_rows = math_topic_tree[math_topic_tree.subject == subject]
        subject_topic_tree = subject_topic_tree.append(subject_rows)
    return subject_topic_tree





def convert_embedding_to_tsne(read_file_affix, method, find_nearest_comparison):
    # The read_file_affix needs to be in the form: fullw10e30
    # for remediation matches, we want: fullw10e30r1
    # Read embedding and token file
    # code needs to run in directory where code lives for root path to work
    root_path = os.path.split(os.getcwd())[0] 
    print('root path: '+ root_path)
    print('read file: '+ read_file_affix)
    print('method: '+ method)
    print('nearest comparison: ' + find_nearest_comparison)
    output_path = root_path + '/'  + 'cahl_output' + '/'
    response_vectors = read_embedding_vectors(create_file_path( output_path, 
                                    'embed_vectors_', read_file_affix))
    response_tokens = read_tokens(create_file_path( output_path,
                                    'embed_index_', read_file_affix))
    learning_state_vectors = read_embedding_vectors(create_file_path( output_path,                                                  'learning_state_vectors_', read_file_affix))
    learning_state_tokens = read_tokens(create_file_path( output_path,                                                               'learning_state_tokens_', read_file_affix))
    # generate tsne file with the
    print('create TSNE')
    tsne_instance = CreateTSNE(response_vectors, learning_state_vectors, response_tokens, learning_state_tokens)
    response_and_learning_tsne_df = tsne_instance.join_response_and_learning_state_df()

    # read remediation matches
    parameters  = method + '_' +  find_nearest_comparison + '_' + read_file_affix + 'r1'
    analysis_path = root_path + '/' + 'cahl_analysis' + '/' + parameters + '/'
    remediation_matches = read_remediation_match(analysis_path +
        'remediation_match_tf') 
    response_and_remediation_tsne_df = tsne_instance.join_response_and_remediation_df(remediation_matches)
    return response_and_learning_tsne_df, response_and_remediation_tsne_df




# The file affix represents specs on the model that we will iterate
# W<Windowsize><TokenLevel><SimilarityApproach>, for examplej# file_affix = 'W10ResponseCosine'

read_file_affix = 'fullw10e30'
method = 'cosine'
find_nearest_comparison = 'response'
response_and_learning_tsne_df, response_and_remediation_tsne_df = convert_embedding_to_tsne(
        read_file_affix, method, find_nearest_comparison)

write_to_tsv(response_and_learning_tsne_df, '~/cahl_output/_response_and_learning_tsne' + read_file_affix )
write_to_tsv(response_and_remediation_tsne_df, '~/cahl_output/_response_and_learning_tsne' + read_file_affix )


selected_subjects = ['algebra',
    # 'algebra-basics',
    'algebra2',
    'ap-calculus-ab',
    # 'ap-calculus-bc',
    # 'ap-statistics',
    'arithmetic',
    # 'basic-geo',
    # 'calculus-1',
    # 'calculus-2',
    # 'cc-1st-grade-math',
    # 'cc-2nd-grade-math',
    # 'cc-eighth-grade-math',
    # 'cc-fifth-grade-math',
    # 'cc-fourth-grade-math',
    # 'cc-kindergarten-math',
    # 'cc-seventh-grade-math',
    # 'cc-sixth-grade-math',
    # 'cc-third-grade-math',
    # 'differential-calculus',
    # 'differential-equations',
    # 'early-math',
    'geometry',
    # 'integral-calculus',
    # 'linear-algebra',
    'pre-algebra',
    # 'precalculus',
    # 'probability',
    # 'statistics-probability',
    'trigonometry']


response_tsne_to_write = join_tsne_to_topic_tree(
                            selected_subjects = selected_subjects,
                            tsne_df = response_and_learning_tsne_df,
                            tsne_join_column = 'exercise' )
write_to_tsv(response_tsne_to_write, '~/cahl_output/joined_response_tsne' + read_file_affix )

remediation_tsne_to_write = join_tsne_to_topic_tree(
                            selected_subjects = selected_subjects,
                            tsne_df = response_and_remediation_tsne_df,
                            tsne_join_column = 'target_exercise' )
write_to_tsv(remediation_tsne_to_write , '~/cahl_output/joined_remediation_tsne' 
        + method +  find_nearest_comparison + read_file_affix  )


