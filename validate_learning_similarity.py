import ast
import csv
import json
import numpy as np
import os
import pdb

'''
Script to validate the output against the manually entered prerequisites

'''


def read_prerequisite_data(file_name, is_json_file=True):
    '''
        read in the prerequisite data
        as a dictionary with the name of each target exercise
        as the key and the list of prerequisite exercises as an array
        example: {'multiplication':['addition','counting']}
    '''
    if is_json_file:
        extension = '.json'
    else:
        extension = 'csv'

    path = os.path.expanduser(file_name+extension)
    reader = open(path,'r')

    if is_json_file:
        prerequisites = json.load(reader)
    else:
        prerequisites = load_prereq_csv_to_dictionary(reader)
    return prerequisites


def load_prereq_csv_to_dictionary(reader):
    prerequisites = {}
    for line in reader:
        # split line by comma delimitation
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

        # [TODO] redo so we store similar tokens by problem type
        #    rather than storing by exercise
        #    we can then find the accuracy rate on a learning token level
        #    rather than on the exercise level

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
        exercise_key = splitline[0]
        # evaluate the set of matches as an array
        similar_exercise = ast.literal_eval(splitline[1])
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
    '''
        Create a precision and recall metric
        For each model match token (by problem type)

        [TODO]
        Precision: did we identify the right prerequisite in the options?
        Token Recall: what % of the right prerequisites were predicted?
            (on a token level)
        Exercise Recall: on the exercise level

    '''
    model_accuracy_output = {}
    for item in model_matches:
        # selected exercise
        exercise = item.split('|')[0]

        try:
            true_prereqs = prerequisites[exercise]
        except KeyError:
            continue

        predicted_prereqs = model_matches[item]
        # determine which of the true prereqs are matched
        # and at what levels they are matched
        is_recall_match, first_match_level, is_precision_match = return_match_logic(
                true_prereqs, predicted_prereqs)
        is_item_match = max(is_recall_match)
        # iterate through predicted prereqs predicted_prereqs
        # if any of the prereqs match
        if exercise not in model_accuracy_output:
            model_accuracy_output[exercise] = {}
            model_accuracy_output[exercise]['is_any_match'] = max(is_recall_match)
        else:
            any_match = model_accuracy_output[exercise]['is_any_match']
            model_accuracy_output[exercise]['is_any_match'] = max(any_match,
                                                                is_item_match)
        model_accuracy_output[exercise][item] = {}
        model_accuracy_output[exercise][item]['precision'] = np.mean(is_precision_match)
        model_accuracy_output[exercise][item]['recall'] = np.mean(is_recall_match)
        model_accuracy_output[exercise][item]['true_prereqs'] = true_prereqs
        model_accuracy_output[exercise][item]['predicted_prereq'] = predicted_prereqs
        model_accuracy_output[exercise][item]['match_level'] = first_match_level
    return model_accuracy_output



def return_match_logic(true_prereqs, predicted_prereqs):
    '''
        see if there are any matches in the predicted
        is_precision_match has an entry for each predicted prereqs
        is_recall_match has an entry for each true prereqs and
        recoreds whether each one matched with any of the predicted exercises
    '''
    # transform predicted problem type prereqs to exercise level
    predicted_exercises = [ prereq.split('|')[0]
            for prereq in predicted_prereqs ]
    is_match = []
    match_levels = []
    is_recall_match, match_levels = return_recall_match(true_prereqs, predicted_exercises)
    is_precision_match = return_precision_match(true_prereqs, predicted_exercises)
    if len(match_levels)>0:
        first_match_level = min(match_levels)
    else:
        first_match_level = None
    return is_recall_match, first_match_level, is_precision_match


def return_precision_match(true_prereqs, predicted_exercises):
    '''
        what is the match rate for the predicted prereqs
        out of the total number of matches per learning token
        on average what % are matched? 
        matches on the exercise level
    '''
    is_precision_match = []
    for i, predicted_exercise in enumerate(predicted_exercises):
        # Iterate through the predicted exercises
        is_predicted_prereq_match = max([predicted_exercise in 
                prereq for prereq in true_prereqs])
        is_precision_match.append(is_predicted_prereq_match)
    return is_precision_match



def return_recall_match(true_prereqs, predicted_exercises):
    '''
       what is the match rate for true prereqs
       out of the total prerequisites in the prerequisite ladder
       how many return a mtching token
       matches on the exercise level
    '''
    is_recall_match = []
    match_levels = []
    for i,level in enumerate(true_prereqs):
        # Iterate through true prerequisites and for each
        for true_prereq in level:
            is_true_prereq_match = true_prereq in predicted_exercises
            is_recall_match.append(is_true_prereq_match)
            if is_true_prereq_match: match_levels.append(i)
    return is_recall_match, match_levels



def print_and_output_sample(analysis_path, model_accuracy_output, affix = '' ):
    '''
        print sample and output
    '''
    precision = calculate_accuracy_rate(model_accuracy_output, 'precision')
    recall = calculate_accuracy_rate(model_accuracy_output, 'recall')
    avg_match_level = calculate_accuracy_rate(model_accuracy_output, 'match_level')
    exercise_recall = calculate_exercise_prereq_recall(model_accuracy_output)
    write_accuracy_output_file(analysis_path,
                file_name = '_accuracy_' + affix,
                precision = precision,
                recall = recall,
                exercise_recall = exercise_recall,
                match_level = avg_match_level)
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



def calculate_accuracy_rate(model_accuracy_output, accuracy_col):
    '''
        calculate the mean accuracy rate
        can be used to calculate avg precision or recall rate
    '''
    accuracy_outputs = []
    for exercise in model_accuracy_output:
        for item in model_accuracy_output[exercise]:
            if item!='is_any_match':
                accuracy = model_accuracy_output[exercise][item][accuracy_col]
                # only append if entry is not None
                if accuracy!=None: accuracy_outputs.append(accuracy)
    return np.mean(accuracy_outputs)


def calculate_exercise_prereq_recall(model_accuracy_output):
    '''
        calculate the mean accuracy rate
    '''
    exercise_recall = []
    for exercise in model_accuracy_output:
        is_match = model_accuracy_output[exercise]['is_any_match']
        exercise_recall.append(np.mean( is_match ))
    return np.mean(exercise_recall)




def write_accuracy_output_file(analysis_path, file_name, precision, recall,
    exercise_recall, match_level):
    '''
        write the random sample as an output file
    '''
    path = os.path.expanduser(analysis_path + file_name+'.csv')
    print(path)
    open_file = open(path, "w")
    with open_file:
        csvwriter = csv.writer(open_file, delimiter = ',')
        csvwriter.writerow(['precision by response',
                'recall by response', 'recall by exercise',
                'avg level of matched prereq'])
        csvwriter.writerow([precision, recall, exercise_recall, match_level])



def output_random_sample(model_accuracy_output, max_sample, is_true_match):
    '''
        randomly sample a series of exercises and print the output
    '''
    items = []
    for exercise in model_accuracy_output:
        for item in model_accuracy_output[exercise]:
            if ( item!='is_any_match'
                and (model_accuracy_output[exercise][item]['precision'] >
                    0)==is_true_match ):
                items.append(item)
    max_size = min(len(items), max_sample)
    sample_items = np.random.choice(items, size = max_size, replace=False)
    sample_output = []
    for item in sample_items:
        exercise = item.split('|')[0]
        sample_output.append(
                [item,
                model_accuracy_output[exercise][item]['true_prereqs'],
                model_accuracy_output[exercise][item]['predicted_prereq'],
                model_accuracy_output[exercise][item]['recall'],
                model_accuracy_output[exercise][item]['match_level']]
            )
    return sample_output




def write_sample_output_file(analysis_path, file_name, output):
    '''
        write the random sample as an output file
        with tab delimited since there may be arrays written to columns
    '''
    path = os.path.expanduser(analysis_path + file_name +'.tsv')
    print(path)
    open_file = open(path, "w")
    with open_file:
        csvwriter = csv.writer(open_file, delimiter = '\t')
        csvwriter.writerow(['item','true prerequisite','model prediction','recall','match_level'])
        csvwriter.writerows(output)



def test_check_model_accuracy():
    '''
        check model accuracy
    '''
    test_model_matches = {'addition_2|type_1':['subtraction_2|type1|true','subtaction_1|type2|true'],
            'addition_2|type_0':['addition_1|type0|false','addition_0|type1|false']}
    test_true_prerequisites = {
            'addition_2':[['addition_1'],['counting']]
            }
    test_model_accuracy_output = check_model_accuracy(test_true_prerequisites, test_model_matches)

    print(test_model_accuracy_output)
    assert test_model_accuracy_output['addition_2']['addition_2|type_1']['recall'] == 0.0
    assert test_model_accuracy_output['addition_2']['addition_2|type_1']['precision'] == 0.0
    assert test_model_accuracy_output['addition_2']['addition_2|type_0']['recall'] == 0.5
    assert test_model_accuracy_output['addition_2']['addition_2|type_0']['precision'] == 0.5
    print('PASSES TEST!!')



def main(read_file_affix, method, find_nearest_comparison, remediation_sample_number):
    '''
        create the nearest comparison
    '''
    path_affix = method + '_' + find_nearest_comparison + '_' + read_file_affix + str(remediation_sample_number)
    root_path = os.path.split(os.getcwd())[0] + '/'
    analysis_path = root_path + 'cahl_analysis' + '/' + path_affix + '/'
    code_path = root_path + 'cahl_remediation_research' + '/'


    ####################################
    prerequisites = read_prerequisite_data('multilevel_prerequisites')
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


if __name__ == "__main__":
    read_file_affix = 'full'
    method = 'cosine'
    find_nearest_comparison = 'response' # response, learn (True-False)
    remediation_sample_number = 10
    main(read_file_affix, method, find_nearest_comparison, remediation_sample_number)



