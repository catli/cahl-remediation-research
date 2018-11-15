import os
import numpy as np
import json
import pdb

# The current prerequisite list derives prerequisites only one level deep
# but we want to look at prerequisites many levels deep
# so we want to create a list of prerequisites that go deeper


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



def create_multilevel_prerequisite(first_level_prerequisites):
    '''
        tranform the one level deep of prerequisites
        input: a dictionary with the first level-prerequisites for each exercise
        output: a dictionary with all levels prerequisites for each exercise
            up to 20th level
    '''
    multilevel_prereq_dictionary = {}

    for exercise in first_level_prerequisites:
        # create first level
        first_level_prereqs = first_level_prerequisites[exercise]
        multilevel_prereq_dictionary[exercise] = []
        multilevel_prereq_dictionary[exercise].append(first_level_prereqs)
        last_level_prereqs = first_level_prereqs
        # loop through 20 levels of prerequisites
        loop = True
        level = 1
        while loop and level<=20:
            next_level_prereqs = extract_the_most_recent_prerequistes(
                first_level_prerequisites = first_level_prerequisites,
                last_level_prereqs = last_level_prereqs)
            if len(next_level_prereqs) == 0:
                loop = False
            else:
                multilevel_prereq_dictionary[exercise].append(next_level_prereqs)
                last_level_prereqs = next_level_prereqs
                level += 1
    return multilevel_prereq_dictionary


def extract_the_most_recent_prerequistes(first_level_prerequisites,  last_level_prereqs):
    '''
        input the most recent prerequisites
        return the next level prerequisites
    '''
    next_level_prerequisites = []
    for exercise in last_level_prereqs:
        try:
            prereqs_for_exercise = first_level_prerequisites[exercise]
            for exercise in prereqs_for_exercise:
                next_level_prerequisites.append(exercise)
        except KeyError:
            continue
    next_level_prerequisites_unique = [ prereq for prereq in np.unique(next_level_prerequisites) ]
    return next_level_prerequisites_unique



def write_output(root_path, multilevel_prerequisites):
    # write the output
    code_path = root_path + 'cahl_remediation_research' + '/'
    # this file should have the same number of state and be in the same order as above
    write_prerequisite_file(path = code_path,
           file_name = 'multilevel_prerequisites',
           multilevel_prerequisites = multilevel_prerequisites)



def write_prerequisite_file(path, file_name, multilevel_prerequisites):
    '''write tokens into flat file, can handle a tuple of tokens per line for similarity tokens'''
    path = os.path.expanduser(path+file_name+'.json')
    print(path)
    open_file = open(path, "w")
    with open_file:
        open_file.write(json.dumps(multilevel_prerequisites))




root_path = os.path.split(os.getcwd())[0] + '/'
first_level_prerequisites = read_prerequisite_data('prerequisites')
multilevel_prerequisites = create_multilevel_prerequisite(first_level_prerequisites)
write_output(root_path, multilevel_prerequisites)
