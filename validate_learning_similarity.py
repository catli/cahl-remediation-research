import os
import numpy as np
import pdb
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


def read_learning_similarity_data(file_name):
    '''
        read in the similarity data
        as a dictionary with the name of each target exercise
        as the key and the list of potential prerequisite exercises
        along with the problem type
    '''
    # Notes for steps:
    # read in each line for the learning similarity data
    # [TODO] update path for CAHL directory
    path = os.path.expanduser('~/Documents/cahl_analysis/'+file_name+'.csv')
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


prerequisites = read_prerequisite_data('prerequisites')
learning_match = read_learning_similarity_data('learning_similar_tokens')
accuracy, true_exercises, false_exercises, same_exercises = check_model_accuracy(prerequisites, learning_match)



# Printout a sample of True and False exercises
true_sample = np.random.choice(true_exercises, size = 10, replace=False)


print('<<<<<<<<<<TRUE SAMPLE>>>>>>>>>>>>>')
for exercise in true_sample:
    print('***EXERCISE***')
    print(exercise)
    print('******prerequisites***')
    print(prerequisites[exercise])
    print('******model matches***')
    print(learning_match[exercise])

false_sample = np.random.choice(false_exercises, size = 10, replace=False)
print('<<<<<<<<<<FALSE SAMPLE>>>>>>>>>>>>>')
for exercise in false_sample:
    print('***EXERCISE***')
    print(exercise)
    print('******prerequisites***')
    print(prerequisites[exercise])
    print('******model matches***')
    print(learning_match[exercise])


pdb.set_trace()

# Read the topic / tutorial for each content


# and join the to the prerequisites / learning token data 


# Read the learning similar tokens
# Read in as dictionary



# Compare the model-generated learning dictionary
# against the prerequisite dictionary
# Store the validation
#    TRUE if matches prerequisites
#    FALSE if  does not match




# Show some example of mach and mismatch
