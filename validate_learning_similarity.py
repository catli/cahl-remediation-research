import os
import numpy as np
import csv
import ast
import pdb

'''
Script to validate the output against the manually entered prerequisites

'''


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
    path = os.path.expanduser(file_path+file_name+'.tsv')
    reader = open(path,'r')
    learning_similarity_exercises = {}
    for line in reader:
        splitline = line.strip().split("\t")
        # split the text by "|" to eliminate problem type, keeping only
        # name of exercise
        exercise_key = splitline[0].split('|')[0]
        # evaluate the set of matches as an array
        similar_exercise = ast.literal_eval(splitline[1])
        try:
            if exercise_key in learning_similarity_exercises:
                [learning_similarity_exercises[exercise_key].append(exercise) for exercise
                    in similar_exercise]
            else:
                # if multiple matches created for the same exercise
                learning_similarity_exercises[exercise_key] = similar_exercise
        except:
            learning_similarity_exercises[exercise_key] = similar_exercise
            pdb.set_trace()
        #     learning_similarity_exercises[exercise_key] = similar_exercise
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
    # accuracy = []
    # avg_matches = []
    # true_exercises = []
    # false_exercises = []
    # same_exercises = []
    model_accuracy_output = {}
    for exercise in prerequisites:
        try:
            remediation_match = model_matches[exercise]
        except KeyError:
            continue
        match = False
        exercise_prereqs = prerequisites[exercise]
        is_matches =[]
        # for prereq in exercise_prereqs:
        for item in remediation_match:
            is_match = [ prereq == item.split("|")[0]
                for prereq in exercise_prereqs]
            any_match = max(is_match)
            match = ( any_match if any_match else match)
            is_matches.append(any_match)
        avg_match = np.mean(is_matches)
        model_accuracy_output[exercise] = {}
        # if remediation_match == exercise: same_exercises.append(exercise)
        #     model_accuracy_output[exercise]['same_exercise'] = True
        model_accuracy_output[exercise]['avg_match'] = avg_match
        model_accuracy_output[exercise]['is_match'] = match
        model_accuracy_output[exercise]['true_prerequisite'] = exercise_prereqs
        model_accuracy_output[exercise]['remediation_match'] = remediation_match
        # if exercise =='model-with-one-step-equations-and-solve':
        #     pdb.set_trace()
        # # exercise_key.append(exercise)
        # if match: true_exercises.append(exercise)
        # if not match: false_exercises.append(exercise)
        # if remediation_match == exercise: same_exercises.append(exercise)
    # return accuracy, avg_matches, true_exercises, false_exercises, same_exercises
    return model_accuracy_output



def calculate_accuracy_rate(model_accuracy_output, accuracy_col):
    '''
        calculate the mean accuracy rate
    '''
    accuracy_outputs = [model_accuracy_output[exercise][accuracy_col]
                            for exercise in model_accuracy_output]
    return np.mean(accuracy_outputs)


def output_random_sample(model_accuracy_output, max_sample, is_true_match):
    '''
        randomly sample a series of exercises and print the output
    '''
    exercises = [exercise for exercise in model_accuracy_output
                    if model_accuracy_output[exercise]['is_match'] == is_true_match]
    max_size = min(len(exercises), max_sample)
    sample_exercises = np.random.choice(exercises, size = max_size, replace=False)
    sample_output = []
    for exercise in sample_exercises:
        sample_output.append(
                [exercise,
                model_accuracy_output[exercise]['true_prerequisite'],
                model_accuracy_output[exercise]['remediation_match'],
                model_accuracy_output[exercise]['avg_match']]
            )
    return sample_output




# def output_random_sample(exercises, max_sample, prerequisites, remediation_match):
#     '''
#         randomly sample a series of exercises and print the output
#     '''
#     max_size = min(len(exercises), max_sample)
#     sample_exercises = np.random.choice(exercises, size = max_size, replace=False)
#     sample_output = []
#     for exercise in sample_exercises:
#         sample_output.append(
#             [exercise,prerequisites[exercise],remediation_match[exercise]])
#     return sample_output




def write_sample_output_file(analysis_path, file_name, output):
    '''
        write the random sample as an output file
    '''
    path = os.path.expanduser(analysis_path + file_name +'.csv')
    print(path)
    open_file = open(path, "w")
    with open_file:
        csvwriter = csv.writer(open_file, delimiter = ',')
        csvwriter.writerow(['exercise','true prerequisite','model matches','avg_match'])
        csvwriter.writerows(output)



def write_accuracy_output_file(analysis_path, file_name, avg_matches, accuracy):
    '''
        write the random sample as an output file
    '''
    path = os.path.expanduser(analysis_path + file_name+'.csv')
    print(path)
    open_file = open(path, "w")
    with open_file:
        csvwriter = csv.writer(open_file, delimiter = ',')
        csvwriter.writerow(['Avg rate of matching','% with any match' ])
        csvwriter.writerow([avg_matches,accuracy])



def print_and_output_sample(analysis_path, model_accuracy_output, affix = '' ):
    '''
        print sample and output
    '''
    avg_match = calculate_accuracy_rate(model_accuracy_output, 'avg_match')
    is_match = calculate_accuracy_rate(model_accuracy_output, 'is_match')
    write_accuracy_output_file(analysis_path,
                file_name = '_accuracy_' + affix,
                avg_matches = avg_match,
                accuracy = is_match)
    # Printout a sample of True and False exercises
    true_sample = output_random_sample(
                        model_accuracy_output = model_accuracy_output,
                        max_sample = 10,
                        is_true_match = True)
    write_sample_output_file( analysis_path,
                file_name = '_true_sample_'+ affix,
                output = true_sample)

    false_sample = output_random_sample(
                        model_accuracy_output = model_accuracy_output,
                        max_sample = 10,
                        is_true_match = False)
    write_sample_output_file(analysis_path,
                file_name =  '_false_sample_'+ affix,
                output = false_sample)






root_path = os.path.split(os.getcwd())[0] + '/'
analysis_path = root_path + 'cahl_analysis' + '/'
code_path = root_path + 'cahl_remediation_research' + '/'


####################################
prerequisites = read_prerequisite_data('prerequisites')
remediation_match = read_learning_similarity_data('remediation_match_tokens',analysis_path)
model_accuracy_output = check_model_accuracy(prerequisites, remediation_match)
print_and_output_sample(analysis_path = analysis_path,
                        model_accuracy_output = model_accuracy_output)

subject_exercises = find_exercise_for_specific_subject(
                        subject = 'pre-algebra',
                        topic_file_name = 'math_topic_tree',
                        topic_file_path = code_path)
subject_prerequisites = filter_prerequisite_in_subject(prerequisites, subject_exercises)
subject_model_accuracy_output = check_model_accuracy(prerequisites= subject_prerequisites,
                                                        model_matches = remediation_match)
print_and_output_sample(analysis_path = analysis_path,
                        model_accuracy_output = subject_model_accuracy_output,
                        affix = 'prealgebra')




