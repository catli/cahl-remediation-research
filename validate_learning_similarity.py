import os
import numpy as np
import pdb
import csv
# Script to validate the output against the manually entered prerequisites
# (1) What percentage of the generated output was also represented in
# the prerequisites list
# (2) What percentage of the generated topic /tutorial was represented in
#    the prerequisites
# (3) Show some examples of right and wrong predictions




# Read the prerequisites data as a dictionary

def read_prerequisite_data(file_name):
    '''
        read in the prerequisite data
        as a dictionary with the name of each target exercise
        as the key and the list of prerequisite exercises as an array
        example: {'multiplication':['addition','counting']}
    '''
    path = os.path.expanduser(file_name+'.csv')
    reader = open(path,'r')
    prerequisites = {}
    for line in reader:
        # split line by comma delimitation
        # [TODO] do we need to split line on space as well as comma
        splitline = line.strip().split(",")
        try:
            prerequisites[splitline[0]].append(splitline[1])
        except KeyError:
            # if key in dictionary not already created
            prerequisites[splitline[0]] = [ splitline[1]]
    return prerequisites


def read_learning_similarity_data(file_name, file_path):
    '''
        read in the similarity data
        as a dictionary with the name of each target exercise
        as the key and the list of potential prerequisite exercises
        along with the problem type
    '''
    # Notes for steps:
    # read in each line for the learning similarity data
    path = os.path.expanduser(file_path+file_name+'.csv')
    reader = open(path,'r')
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


def find_exercise_for_specific_subject(subject,
        topic_file_path, topic_file_name):
    '''
        read in the topic tree and filter for exercises of one subject
        that way we can focus on patterns for one subject
    '''
    path = os.path.expanduser(topic_file_path+topic_file_name+'.csv')
    reader = open(path,'r')
    topic_exercises = []
    for line in reader:
        splitline = line.strip().split(",")
        if subject in splitline:
            # if subject found in the line, then append
            topic_exercises.append(splitline[0])
    return np.unique(topic_exercises)



def filter_prerequisite_in_subject(prerequisites, subject_exercises):
    '''
        keep only the prerequisites that are in the list of exercises
        in a subject subset
    '''
    filtered_prerequisites = {}
    for exercise in prerequisites:
        if exercise in subject_exercises:
            filtered_prerequisites[exercise] = prerequisites[exercise]
    return filtered_prerequisites


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


def check_model_accuracy(prerequisites, model_matches):
    # [TODO] match the accuracy against
    # exercise_key = []
    accuracy = []
    true_exercises = []
    false_exercises = []
    same_exercises = []
    for exercise in prerequisites:
        try:
            remediation_match = model_matches[exercise]
        except KeyError:
            continue
        match = False
        exercise_prereqs = prerequisites[exercise]
        for prereq in exercise_prereqs:
            if prereq in [remediation_match]:
                match = True
        accuracy.append(match)
        # exercise_key.append(exercise)
        if match: true_exercises.append(exercise)
        if not match: false_exercises.append(exercise)
        if remediation_match == exercise: same_exercises.append(exercise)
    return accuracy, true_exercises, false_exercises, same_exercises


def write_output_file(file_name, output):
    '''
        write the random sample as an output file
    '''
    path = os.path.expanduser(file_name+'.csv')
    print(path)
    open_file = open(path, "w")
    with open_file:
        csvwriter = csv.writer(open_file, delimiter = ',')
        csvwriter.writerows(output)


def output_random_sample(exercises, max_sample, prerequisites, remediation_match):
    '''
        randomly sample a series of exercises and rint the output
    '''
    max_size = min(len(exercises), max_sample)
    sample_exercises = np.random.choice(exercises, size = max_size, replace=False)
    sample_output = []
    for exercise in sample_exercises:
        sample_output.append(
            [exercise,prerequisites[exercise],remediation_match[exercise]])
    return sample_output






prerequisites = read_prerequisite_data('prerequisites')
# [TODO] update path for CAHL directory
remediation_match = read_learning_similarity_data('remediation_match_tokens',
                    '~/Documents/cahl_analysis/')
accuracy, true_exercises, false_exercises, same_exercises = check_model_accuracy(prerequisites, remediation_match)


# [TODO] update path for CAHL directory
# subject_exercises = find_exercise_for_specific_subject(
#                         subject = 'pre-algebra',
#                         topic_file_name = 'math_topic_tree',
#                         topic_file_path = '~/Documents/cahl_remediation_research/')
# subject_prerequisites = filter_prerequisite_in_subject(prerequisites, subject_exercises)
# print(subject_prerequisites)
# accuracy, true_exercises, false_exercises, same_exercises = check_model_accuracy(subject_prerequisites, remediation_match)


# Printout a sample of True and False exercises
true_sample = output_random_sample(exercises = true_exercises,
                    max_sample = 10,
                    prerequisites = prerequisites,
                    remediation_match= remediation_match)
write_output_file('true_sample',true_sample)

false_sample = output_random_sample(exercises = false_exercises,
                    max_sample = 10,
                    prerequisites = prerequisites,
                    remediation_match= remediation_match)
write_output_file('false_sample',false_sample)



pdb.set_trace()


