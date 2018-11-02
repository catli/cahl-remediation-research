############################################
# [2] Visualize using t-NSE, a tool to visualize high dimensional data
# by converting to join probabilities and minimizing divergence
# we minimize it to an embedding with 2 dimensions  (n_components = 2)
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import pdb



##########################################
# Functions to read and write files

def read_embedding_vectors(file_name):
    '''
        read in the response embedding vector files
        these were stored using np.savetxt (.out)
        return a matrix of embedding vectors
        # [TESTING] before implementing, change to cal.berkeley.edu directory
    '''
    path = os.path.expanduser('~/Documents/cahl_output/'+file_name+'.out')
    response_vectors = np.loadtxt(path, delimiter = ',')
    return response_vectors


def read_tokens(file_name):
    '''
        read in the flat file containing the list of response tokens
        return an array of response tokens associated with embedding vectors
        # [TESTING] before implementing, change to cal.berkeley.edu directory
    '''
    path = os.path.expanduser('~/Documents/cahl_output/'+file_name+'.csv')
    reader = open(path,'r')
    response_tokens = []
    for line in reader:
        response_tokens.append(line.strip())
    return response_tokens


def read_remediation_tokens(file_name):
    '''
        read in the flat file containing the list of response tokens
        return an array of response tokens associated with embedding vectors
        # [TESTING] before implementing, change to cal.berkeley.edu directory
    '''
    path = os.path.expanduser('~/Documents/cahl_analysis/'+file_name+'.csv')
    reader = open(path,'r')
    remediation_tokens = {}
    for line in reader:
        # delimit by comma
        splitline = line.strip().split(',')
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



def store_exercise_for_specific_subject(subject,
        topic_file_path, topic_file_name):
    '''
        read in the topic tree and filter for exercises of one subject
        that way we can focus on patterns for one subject
    '''
    path = os.path.expanduser(topic_file_path+topic_file_name+'.csv')
    reader = open(path,'r')
    topic_exercises = {}
    for line in reader:
        splitline = line.strip().split(",")
        exercise = splitline[0]
        unit = splitline[2]
        is_in_splitline = subject in splitline
        is_in_topic_exercise = exercise in topic_exercises
        if is_in_splitline and not is_in_topic_exercise:
            # if subject found in the line, then append
            topic_exercises[exercise] = unit
    return topic_exercises



def filter_embedding_vectors_in_subject(tokens, vectors, subject_exercises):
    '''
        create a new embedding and token vector
        store the tokens and for each one
        as well as the topic and tutorials
    '''
    subject_tokens = []
    subject_vectors = []
    subject_topics = []
    for i, token in enumerate(tokens):
        exercise = token.split('|')[0]
        if exercise in subject_exercises:
            subject_tokens.append(token)
            subject_vectors.append(vectors[i])
            subject_topics.append(subject_exercises[exercise])
    return subject_tokens, subject_vectors, subject_topics


def plt_tsne_for_subject(tsne_vectors, tokens, subject, file_affix):
    '''
        for selected subject and associated tokens
        plot the generated embedding vectors
        with a color code for unit
    '''
    subject_exercises = store_exercise_for_specific_subject(
                            subject = subject,
                            topic_file_name = 'math_topic_tree',
                            topic_file_path = '~/Documents/cahl_remediation_research/')
    # Filter out embedding for specific subject
    subject_tokens, subject_tsne_vectors, subject_topics = filter_embedding_vectors_in_subject(
                                                tokens, tsne_vectors, subject_exercises)
    fig, ax = plt.subplots(figsize=(15,15))
    for topic in np.unique(subject_topics):
        topic_loc = [i for i,s in enumerate(subject_topics) if s==topic]
        # find the tsne vectors in this subject
        tsne_0 = [subject_tsne_vectors[loc][0] for loc in topic_loc]
        tsne_1 = [subject_tsne_vectors[loc][1] for loc in topic_loc]
        ax.plot(tsne_0, tsne_1, '.', markersize= 10, label = topic)
    ax.legend()
    file_name = 'tsne_'+ subject + '_' + file_affix + '.jpg'
    plt.savefig(file_name)


def find_remediation_embedding(target_token,
            tokens, remediation_matches, vectors):
    '''
        for target token, find the remediation token
        strip the response from target token before
        and the remediation vectors (TSNE vector)
    '''
    target_problem_type = '|'.join(target_token.split('|')[:-1])
    remediation_token  = remediation_matches[target_problem_type]
    remediation_ind = tokens.index(remediation_token)
    remediation_vector = vectors[remediation_ind]
    return remediation_vector


def plt_remediation_tsne_for_subject(tsne_vectors, tokens, subject,
    remediation_matches, file_affix):
    '''
        for selected subject and associated tokens
        plot the generated embedding vectors
        with a color code for unit
    '''
    subject_exercises = store_exercise_for_specific_subject(
                            subject = subject,
                            topic_file_name = 'math_topic_tree',
                            topic_file_path = '~/Documents/cahl_remediation_research/')
    # Filter out embedding for specific subject
    subject_tokens, subject_tsne_vectors, subject_topics = filter_embedding_vectors_in_subject(
                                                tokens, tsne_vectors, subject_exercises)
    fig, ax = plt.subplots(figsize=(15,15))
    # [TODO] try plt.arrow
    for topic in np.unique(subject_topics):
        topic_loc = [i for i,s in enumerate(subject_topics) if s==topic]
        for i, token in enumerate(subject_tokens):
            # find the remediation vector associated with current token
            # topic_loc = [i for i,s in enumerate(subject_topics) if s==topic]
            # # find the tsne vectors in this subject
            # tsne_0 = [subject_tsne_vectors[loc][0] for loc in topic_loc]
            # tsne_1 = [subject_tsne_vectors[loc][1] for loc in topic_loc]
            # # plot the dots for each subject
            # ax.plot(tsne_0, tsne_1, '.', markersize= 10, label = topic)

            target_vector = subject_tsne_vectors[i]
            try:
                # [TODO] how to plot the label color without
                # generated a separate legend for each observation
                remediation_vector = find_remediation_embedding(
                    target_token = token,
                    tokens = tokens,
                    remediation_matches = remediation_matches,
                    vectors = tsne_vectors)
                ax.plot(target_vector, remediation_vector,
                    color = '#C0C0C0')
            except:
                continue    
    # ax.legend()
    file_name = 'tsne_remediation_'+ subject + '_' + file_affix + '.jpg'
    plt.savefig(file_name)


# TESTING: Change file names
# The file affix represents specs on the model that we will iterate
# W<Windowsize><TokenLevel><SimilarityApproach>, for example
# file_affix = 'W10ResponseCosine'

file_affix = 'W10ResponseEuclidean'

vectors = read_embedding_vectors('embed_vectors_full')
tokens = read_tokens('embed_index_full')
remediation_token_matches = read_remediation_tokens(
    'remediation_match_tokens')
tsne_vectors = TSNE(n_components=2).fit_transform(vectors)

subjects = ['pre-algebra', 'arithmetic','trigonometry',
        'ap-calculus-ab','cc-1st-grade-math','eb-1-primaria']

for subject in subjects:
    plt_tsne_for_subject(tsne_vectors = tsne_vectors,
        tokens = tokens,
        subject = subject,
        file_affix = file_affix)
    plt_remediation_tsne_for_subject(tsne_vectors = tsne_vectors,
        tokens = tokens, 
        subject = subject,
        remediation_matches = remediation_token_matches,
        file_affix = file_affix)