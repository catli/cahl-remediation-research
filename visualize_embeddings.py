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



##########################################
# Functions to read and write files


def read_remediation_tokens(file_name):
    '''
        read in the flat file containing the list of response tokens
        return an array of response tokens associated with embedding vectors
        # [TESTING] before implementing, change to cal.berkeley.edu directory
    '''
    path = os.path.expanduser(file_name+'.tsv')
    reader = open(path,'r')
    remediation_tokens = {}
    for line in reader:
        # delimit by comma
        splitline = line.strip().split('\t')
        # store first column as key and the second column as value
        remediation_tokens[splitline[0]] = splitline[1]
    return remediation_tokens



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
    dataframe.to_csv(file_name,
        sep = '\t', 
        header = True,
        index = False)




##########################################
# Functions to create and join data frame


def join_tsne_to_topic_tree_dataframe(df_tsne, df_topic_tree):
    '''
       join the tsne dataframe tot he subject and topic tree
       shoudl be a many to many join
    '''
    return pd.merge(df_tsne, df_topic_tree,
        left_on = 'exercise', right_on='exercise', how='inner')


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



def create_dataframe_of_tsne_without_remediation(response_tokens, tsne_response_vectors):
    '''
        create a pandas dataset with the x, y coodinates (tsne),
        the token name, exercise name , problem type (if available),
        and correctness of answer if available
    '''
    response_tsne_df = pd.DataFrame(columns = ['x','y','exercise','response','token'])
    for i, token in enumerate(response_tokens):
        vector = tsne_response_vectors[i]
        exercise = token.split('|')[0]
        response_accuracy = token.split('|')[2]
        response_tsne_df = response_tsne_df.append({
                                'x': vector[0], 'y': vector[1],
                                'exercise': exercise,
                                'response': response_accuracy,
                                'token': token }, ignore_index = True)
    return response_tsne_df



def create_dataframe_of_learning_tsne(learning_tokens,
    tsne_learning_vectors, response_tokens, tsne_response_vectors, remediation_matches):
    '''
        create a pandas dataset with the x, y coodinates (tsne),
        the token name, exercise name , problem type (if available),
        and x, y coordinates of remediation_matches
    '''
    learning_tsne_df = pd.DataFrame(columns = ['x','y',
                                'exercise','token',
                                'remediation_token','x2','y2'])
    for i, token in enumerate(learning_tokens):
        learning_vector = tsne_learning_vectors[i]
        remediation_response_token =  ast.literal_eval(
                                        remediation_matches[token])[0]
        try:
            exercise = token.split('|')[0]
            token_index = response_tokens.index(remediation_response_token)
            remediation_vector = tsne_response_vectors[token_index]
        except:
            pdb.set_trace()
        learning_tsne_df = learning_tsne_df.append( {
                                'x': learning_vector[0], 'y': learning_vector[1],
                                'exercise': exercise,
                                'token': token,
                                'remediation_token': remediation_response_token,
                                'x2': remediation_vector[0],
                                'y2': remediation_vector[1]}, ignore_index = True)
    return learning_tsne_df



# def store_exercise_for_specific_subject(subject,
#         topic_file_path, topic_file_name):
#     '''
#         read in the topic tree and filter for exercises of one subject
#         that way we can focus on patterns for one subject
#     '''
#     path = os.path.expanduser(topic_file_path+topic_file_name+'.csv')
#     reader = open(path,'r')
#     topic_exercises = {}
#     for line in reader:
#         splitline = line.strip().split(",")
#         exercise = splitline[0]
#         unit = splitline[2]
#         is_in_splitline = subject in splitline
#         is_in_topic_exercise = exercise in topic_exercises
#         if is_in_splitline and not is_in_topic_exercise:
#             # if subject found in the line, then append
#             topic_exercises[exercise] = unit
#     return topic_exercises




# def filter_embedding_vectors_in_subject(tokens, vectors, subject_exercises):
#     '''
#         create a new embedding and token vector
#         store the tokens and for each one
#         as well as the topic and tutorials
#     '''
#     subject_tokens = []
#     subject_vectors = []
#     subject_topics = []
#     for i, token in enumerate(tokens):
#         exercise = token.split('|')[0]
#         if exercise in subject_exercises:
#             subject_tokens.append(token)
#             subject_vectors.append(vectors[i])
#             subject_topics.append(subject_exercises[exercise])
#     return subject_tokens, subject_vectors, subject_topics




# def plt_tsne_for_subject(tsne_vectors, tokens, subject, file_affix):
#     '''
#         for selected subject and associated tokens
#         plot the generated embedding vectors
#         with a color code for unit
#     '''
#     subject_exercises = store_exercise_for_specific_subject(
#                             subject = subject,
#                             topic_file_name = 'math_topic_tree',
#                             topic_file_path = '~/Documents/cahl_remediation_research/')
#     # Filter out embedding for specific subject
#     subject_tokens, subject_tsne_vectors, subject_topics = filter_embedding_vectors_in_subject(
#                                                 tokens, tsne_vectors, subject_exercises)
#     fig, ax = plt.subplots(figsize=(15,15))
#     for topic in np.unique(subject_topics):
#         topic_loc = [i for i,s in enumerate(subject_topics) if s==topic]
#         # find the tsne vectors in this subject
#         tsne_0 = [subject_tsne_vectors[loc][0] for loc in topic_loc]
#         tsne_1 = [subject_tsne_vectors[loc][1] for loc in topic_loc]
#         ax.plot(tsne_0, tsne_1, '.', markersize= 10, label = topic)
#     ax.legend()
#     file_name = 'tsne_'+ subject + '_' + file_affix + '.jpg'
#     plt.savefig(file_name)



# def plt_remediation_tsne_for_subject(tsne_vectors, tokens, subject,
#     remediation_matches, file_affix):
#     '''
#         for selected subject and associated tokens
#         plot the generated embedding vectors
#         with a color code for unit
#     '''
#     subject_exercises = store_exercise_for_specific_subject(
#                             subject = subject,
#                             topic_file_name = 'math_topic_tree',
#                             topic_file_path = '~/Documents/cahl_remediation_research/')
#     # Filter out embedding for specific subject
#     subject_tokens, subject_tsne_vectors, subject_topics = filter_embedding_vectors_in_subject(
#                                                 tokens, tsne_vectors, subject_exercises)
#     fig, ax = plt.subplots(figsize=(15,15))
#     # [TODO] try plt.arrow
#     for topic in np.unique(subject_topics):
#         topic_loc = [i for i,s in enumerate(subject_topics) if s==topic]
#         for i, token in enumerate(subject_tokens):
#             # find the remediation vector associated with current token
#             # topic_loc = [i for i,s in enumerate(subject_topics) if s==topic]
#             # # find the tsne vectors in this subject
#             # tsne_0 = [subject_tsne_vectors[loc][0] for loc in topic_loc]
#             # tsne_1 = [subject_tsne_vectors[loc][1] for loc in topic_loc]
#             # # plot the dots for each subject
#             # ax.plot(tsne_0, tsne_1, '.', markersize= 10, label = topic)

#             target_vector = subject_tsne_vectors[i]
#             try:
#                 # [TODO] how to plot the label color without
#                 # generated a separate legend for each observation
#                 remediation_vector = find_remediation_embedding(
#                     target_token = token,
#                     tokens = tokens,
#                     remediation_matches = remediation_matches,
#                     vectors = tsne_vectors)
#                 ax.plot(target_vector, remediation_vector,
#                     color = '#C0C0C0')
#             except:
#                 continue    
#     # ax.legend()
#     file_name = 'tsne_remediation_'+ subject + '_' + file_affix + '.jpg'
#     plt.savefig(file_name)


# def find_remediation_embedding(target_token,
#             tokens, remediation_matches, vectors):
#     '''
#         for target token, find the remediation token
#         strip the response from target token before
#         and the remediation vectors (TSNE vector)
#     '''
#     target_problem_type = '|'.join(target_token.split('|')[:-1])
#     remediation_token  = remediation_matches[target_problem_type]
#     remediation_ind = tokens.index(remediation_token)
#     remediation_vector = vectors[remediation_ind]
#     return remediation_vector


# TESTING: Change file names
# The file affix represents specs on the model that we will iterate
# W<Windowsize><TokenLevel><SimilarityApproach>, for example
# file_affix = 'W10ResponseCosine'
from util_functions import read_embedding_vectors
from util_functions import read_tokens




root_path = os.path.split(os.getcwd())[0] + '/'
code_path = root_path + 'cahl_remediation_research' + '/'
output_path = root_path + 'cahl_output' + '/'

analysis_path_affix = 'W10CosineResponseC1' + '/'
analysis_path = root_path + 'cahl_analysis' + '/' + analysis_path_affix

read_file_affix = 'full'
response_vectors = read_embedding_vectors(output_path +
                        'embed_vectors_' +
                        read_file_affix)
response_tokens = read_tokens(output_path +
                        'embed_index_' +
                        read_file_affix)

learning_vectors = read_embedding_vectors(analysis_path +
                        'learning_state_vectors')
learning_tokens = read_tokens(analysis_path +
                        'learning_state_tokens' )

# read the remediation match tokens tokens
remediation_matches = read_remediation_tokens(analysis_path +
        'remediation_match_tokens')




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

math_topic_tree_df = read_topic_tree_data_frame(file_name = code_path + 'math_topic_tree')


tsne_response_vectors = TSNE(n_components=2).fit_transform(response_vectors)
subject_topic_tree_df = create_dataframe_of_topic_tree(
                            subjects = selected_subjects,
                            math_topic_tree = math_topic_tree_df)
response_tsne_df =  create_dataframe_of_tsne_without_remediation(
                            response_tokens = response_tokens,
                            tsne_response_vectors = tsne_response_vectors)
joined_response_tsne = join_tsne_to_topic_tree_dataframe(
                            df_tsne = response_tsne_df, 
                            df_topic_tree = subject_topic_tree_df)
write_to_tsv(joined_response_tsne, 'joined_response_tsne.tsv' )


tsne_learning_vectors = TSNE(n_components=2).fit_transform(learning_vectors)
learning_tsne_df = create_dataframe_of_learning_tsne(
                           learning_tokens = learning_tokens,
                           tsne_learning_vectors = tsne_learning_vectors,
                           response_tokens = response_tokens,
                           tsne_response_vectors = tsne_response_vectors,
                           remediation_matches = remediation_matches)
pdb.set_trace()
joined_learning_tsne = join_tsne_to_topic_tree_dataframe(
                            df_tsne = learning_tsne_df, 
                            df_topic_tree = subject_topic_tree_df)

write_to_tsv(joined_learning_tsne, 'joined_learning_tsne.tsv' )


