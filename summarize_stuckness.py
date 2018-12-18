import time
import csv
import os
import pdb
import json

'''
Script to iterate through the script
'''

class SummarizeStuckness():

    def __init__(self, read_filename, write_filename):
        print('initialize '+ read_filename)
        self.reader = open(read_filename,'r')
        self.writefile = open(write_filename, 'w')
       
    def iterate_through_lines(self, prerequisites, units, lessons, sessions = False):
        '''
            read file and write the lines
            does not use readlines so there's less strain on memory
            it would be great helpful to parallelize this function but
            not sure how to efficiently do this and maintain the sort order 
        '''
        self.last_sha_id = 'sha_id'
        self.last_problem = 'exercise|problem_type'
        self.user_attempts = {}
        self.user_data = {'stuck':{},'unstuck':{}, 'never_stuck':[], 'stuck_correct':{}  }
        self.csvwriter = csv.writer(self.writefile, delimiter = ',') 
        self.write_header()
        next(self.reader)
        # first_line = self.reader.readline()
        counter = 1
        for line in self.reader:
            self.parse_line(line, prerequisites, units, lessons, sessions)
            counter+=1
            if counter % 1000000 == 0:
                print(counter)
    
    def write_header(self):
        self.csvwriter.writerow([ 'sha_id', 
            'total_problems',
            'never_stuck_problems', 
            'never_unstuck_problems',
            'unstuck_problems',
            'reattempted_stuck_problems',
            'unstuck_different_exercise',
            'unstuck_remediation_problems',
            'unstuck_correct_remdiation_problems',
            'unstuck_prereq_avail_problems',
            'unstuck_is_prereq_match_problems',
            'unstuck_is_topic_tree_avail_problems',  
            'unstuck_is_not_unit_match_problems',  
            'unstuck_is_not_lesson_match_problems',
            'unstuck_topic_tree_avail_remediation_items',
            'unstuck_unit_match_remediation_items',  
            'unstuck_lesson_match_remediation_items'
            #[stuck_prereq!]
            #'stuck_prereq_avail',
            #'stuck_prereq_match'
            ])
    
    def parse_line(self, line, prerequisites, units, lessons, sessions=False):
        '''
           Parse through each line and store the values 
        '''
        line_delimited = line.split(',')
        # if sessions = True, then id by session
        if sessions:
            sha_id = line_delimited[0]+line_delimited[2]
        else:
            sha_id = line_delimited[0]
        exercise = line_delimited[5]
        problem_type = line_delimited[7]
        correct = line_delimited[8] == 'true'
        attempt_numbers = int(line_delimited[12])
        problem = exercise + '|' +  problem_type
        if sha_id != self.last_sha_id:
            self.summarize_old_sha_id(prerequisites)
            self.last_sha_id = sha_id
            self.last_problem = problem
            self.user_attempts = {}
            self.update_attempts(correct, attempt_numbers, problem) 
        else:
            self.update_attempts(correct, attempt_numbers, problem)  
            self.add_new_data_for_user( 
                    problem_type, exercise, correct, prerequisites, units, lessons)
            self.last_problem = problem

    def update_attempts(self, correct, attempt_numbers, problem):
        if problem not in self.user_attempts:
            self.user_attempts[problem] = {}
            self.user_attempts[problem]['correct'] = 0
            self.user_attempts[problem]['incorrect'] = 0
        if correct:
            self.user_attempts[problem]['correct']+=1
        else:
            self.user_attempts[problem]['incorrect']+= max(attempt_numbers-1,1)
             

    def summarize_old_sha_id(self, prerequisites):
        '''
            summarize user data 
            and write the array of summary stats
            to the file
        '''
        never_stuck_problems = len(self.user_data['never_stuck'])
        never_unstuck_problems = len(self.user_data['stuck'].keys())
        unstuck_problems = len(self.user_data['unstuck'].keys())
        reattempted_stuck_problems = unstuck_problems + self.count_reattempt_on_stuck()
        # [stuck_prereq!]
        # stuck_prereq_avail, stuck_prereq_match = self.count_prereq_use_on_stuck( prerequisites)
        unstuck_different_exercise = 0
        unstuck_remediation_problems = 0 
        unstuck_correct_remdiation_problems = 0
        unstuck_prereq_avail_problems = 0
        unstuck_is_prereq_match_problems = 0  
        unstuck_is_topic_tree_avail_problems = 0  
        unstuck_is_not_unit_match_problems  = 0  
        unstuck_is_not_lesson_match_problems  = 0  
        unstuck_topic_tree_avail_remediation_items = 0  
        unstuck_unit_match_remediation_items = 0  
        unstuck_lesson_match_remediation_items = 0  
        for unstuck_item in self.user_data['unstuck']:
            unstuck_array = self.user_data['unstuck'][unstuck_item] 
            unstuck_different_exercise += unstuck_array['different_exercise_remediation_problems']
            unstuck_remediation_problems += unstuck_array['remediation_problems']
            unstuck_correct_remdiation_problems += unstuck_array['correct_remediation_problems']
            unstuck_prereq_avail_problems += unstuck_array['is_prereqs_available']  
            unstuck_is_prereq_match_problems += unstuck_array['is_prereqs_match']
            unstuck_is_topic_tree_avail_problems += unstuck_array['is_topic_tree_avail']  
            unstuck_is_not_unit_match_problems += unstuck_array['is_not_unit_match']  
            unstuck_is_not_lesson_match_problems += unstuck_array['is_not_lesson_match'] 
            unstuck_topic_tree_avail_remediation_items += unstuck_array[
                    'topic_tree_avail_items']  
            unstuck_unit_match_remediation_items += unstuck_array['unit_match_items']  
            unstuck_lesson_match_remediation_items += unstuck_array['lesson_match_items']
        self.csvwriter.writerow([ self.last_sha_id, 
            never_unstuck_problems + never_stuck_problems +  unstuck_problems,
            never_stuck_problems, 
            never_unstuck_problems,
            unstuck_problems,
            reattempted_stuck_problems,
            unstuck_different_exercise,
            unstuck_remediation_problems,
            unstuck_correct_remdiation_problems,
            unstuck_prereq_avail_problems,
            unstuck_is_prereq_match_problems,
            unstuck_is_topic_tree_avail_problems,  
            unstuck_is_not_unit_match_problems ,  
            unstuck_is_not_lesson_match_problems , 
            unstuck_topic_tree_avail_remediation_items,
            unstuck_unit_match_remediation_items,  
            unstuck_lesson_match_remediation_items
            #[stuck_prereq!]
            #stuck_prereq_avail, 
            #stuck_prereq_match
            ])
        # clear user data 
        self.user_data = {'stuck':{},'unstuck':{}, 'never_stuck':[], 'stuck_correct':{}  }

    def count_reattempt_on_stuck(self):
        '''
            iterate through all the stuck exercises and see if learner
            reattempted any of these problems
        '''
        reattempted_stuck_problems = 0
        for problem in self.user_data['stuck']:
            if problem in self.user_data['stuck'][problem]:
                reattempted_stuck_problems += 1 
        return  reattempted_stuck_problems

    def count_prereq_use_on_stuck(self, prerequisites):
        '''
            iterate through all the stuck exercises and see if learner
            attempted a prereq for that exercise 
        '''
        stuck_prereq_avail = 0
        stuck_prereq_match = 0
        for problem in self.user_data['stuck']:
            exercise = problem.split('|')[0]
            practice_list = self.user_data['stuck'][problem]
            practice_exercise_list = [item.split('|')[0] for item in practice_list]
            prereq_avail, is_recall_matches= self.check_against_prereqs(problem , prerequisites, practice_exercise_list)
            stuck_prereq_avail += prereq_avail
            stuck_prereq_match += int(max(is_recall_matches))
        return stuck_prereq_avail, stuck_prereq_match  

    def add_new_data_for_user(self, 
            problem_type, exercise, correct, prerequisites, units, lessons):
        problem = exercise + '|' + problem_type 
        if self.user_attempts[problem]['correct']>=2 and problem in self.user_data['stuck']:
            # [TODO] add units and lessons
            self.user_data['unstuck'][problem]  = self.summarize_unstuck(
                    problem, prerequisites, units, lessons)
            del self.user_data['stuck'][problem]
            del self.user_data['stuck_correct'][problem]
        self.add_to_stuck(problem, correct)
        if problem in list(self.user_data['unstuck'].keys()) \
            + self.user_data['never_stuck']:
            pass 
        elif self.user_attempts[problem]['correct']>=2 and \
            problem not in self.user_data['stuck']:
            self.user_data['never_stuck'].append(problem)
        elif self.user_attempts[problem]['incorrect']>=2 and problem not in self.user_data['stuck']:
            self.user_data['stuck'][problem] = []
            self.user_data['stuck_correct'][problem] = []

    def add_to_stuck(self, problem, correct):
        '''
           add the new problem to the list of stuck items 
        '''
        for stuck_item in self.user_data['stuck']:
            self.user_data['stuck'][stuck_item].append(problem)
            self.user_data['stuck_correct'][stuck_item].append(correct)
    
    def summarize_unstuck(self, problem, prerequisites, units, lessons):
        '''
            output: output the unstuck summary stats array
            which lists the attibutes of the unstuckness token
            ['different_exercise','remediation_problems','correct_remediation_problems']
        '''
        exercise = problem.split('|')[0]
        correct_list = self.user_data['stuck_correct'][problem]
        remediation_list = self.user_data['stuck'][problem]
        remediation_exercise_list = [item.split('|')[0] for item in remediation_list]
        unstuck_state = {} 
        unstuck_state['different_exercise_remediation_problems'] = sum(
                [item != exercise for item in remediation_exercise_list]) 
        unstuck_state['remediation_problems'] = len(remediation_list) 
        unstuck_state['correct_remediation_problems'] = sum(correct_list)
        # check against prereqs
        prereq_avail, is_recall_matches = self.check_against_prereqs(
                problem, prerequisites, remediation_exercise_list)
        unstuck_state['is_prereqs_available'] = int(prereq_avail) 
        unstuck_state['is_prereqs_match'] = int(max(is_recall_matches))
        unstuck_state['prereqs_available_items'] = len(is_recall_matches)*int(prereq_avail)
        unstuck_state['prereqs_match_items'] = sum(is_recall_matches)
        # calculate whether remediation exercise match target exercise unit /lessons
        is_topic_tree_avail, is_unit_matches, is_lesson_matches = self.check_topic_tree_match(
                problem, remediation_exercise_list, units, lessons)
        unstuck_state['is_topic_tree_avail'] = int(is_topic_tree_avail) 
        unstuck_state['is_unit_match'] = int(max(is_unit_matches))
        unstuck_state['is_lesson_match'] = int(max(is_lesson_matches))
        unstuck_state['is_not_unit_match'] = ( 1 - int(min(is_unit_matches)))*int(is_topic_tree_avail)
        unstuck_state['is_not_lesson_match'] = (1 - int(min(is_lesson_matches)))*int(is_topic_tree_avail)
        unstuck_state['topic_tree_avail_items'] = len(is_unit_matches)*int(is_topic_tree_avail) 
        unstuck_state['unit_match_items'] = sum(is_unit_matches)
        unstuck_state['lesson_match_items'] = sum(is_lesson_matches)
        return unstuck_state

    def check_against_prereqs(self, problem, prerequisites, remediation_exercise_list): 
        '''
            Check if prerequisites available for an exercise
            If available check if there matches
        '''
        exercise = problem.split('|')[0]
        if exercise in prerequisites:
            prereq_avail = True
            true_prereqs = prerequisites[exercise]
            is_recall_matches = self.return_prereq_match(
                    true_prereqs, remediation_exercise_list)
        else: 
            prereq_avail = False
            is_recall_matches = [False]
            #match_levels = 0
        return prereq_avail, is_recall_matches #, match_levels

    def return_prereq_match(self, true_prereqs, remediation_exercise_list):
        '''
            Find the prerequisites of the target exercise being worked on 
            and see if any items in the remediation list is a prerequisite
        '''
        is_recall_matches = []
        for i,level in enumerate(true_prereqs):
            # Iterate through true prerequisites and for each
            for true_prereq in level:
                is_true_prereq_match = true_prereq in remediation_exercise_list
                is_recall_matches.append(is_true_prereq_match)
        return is_recall_matches

    def check_topic_tree_match(self, problem, remediation_exercise_list, units, lessons): 
        '''
            check if the if remediation exercises are in the same unit
            or lessons as target exercise
        '''
        exercise = problem.split('|')[0]
        if exercise in units and remediation_exercise_list:
            is_topic_tree_avail = True 
            is_unit_matches = self.return_topic_tree_match(
                    exercise, remediation_exercise_list, units)
            is_lesson_matches = self.return_topic_tree_match(
                    exercise, remediation_exercise_list, lessons)
        else: 
            is_topic_tree_avail = False
            is_unit_matches = [False]
            is_lesson_matches  = [False]
        return is_topic_tree_avail, is_unit_matches, is_lesson_matches  

    def return_topic_tree_match(self, 
            exercise, remediation_exercise_list, match_dict):
        target_match = set(match_dict[exercise])
        is_topic_tree_matches = []
        for item in remediation_exercise_list:
            remediation_match = set(match_dict[item])
            is_match = len(target_match.intersection(
                        remediation_match))>0
            is_topic_tree_matches.append(is_match)
        return  is_topic_tree_matches


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

def read_topic_tree_data(file_name):
    '''
        read in the topic tree data
        as a dictionary with the name of each target exercise
        as the key and the list of units and tops an array
        example: {'multiplication':['1st-grade-math-topic-1',
        'ny-engage-math-topic']}
    '''
    path = os.path.expanduser(file_name+'.csv')
    reader = open(path,'r')
    units, lessons = load_topic_tree_to_dict(reader)
    return units, lessons


def load_topic_tree_to_dict(topic_tree_reader):
    units = {}
    lessons = {}
    for line in topic_tree_reader:
        # split line by comma delimitation
        splitline = line.strip().split(",")
        try:
            units[splitline[0]].append(splitline[2])
            lessons[splitline[0]].append(splitline[3])
        except KeyError:
            # if key in dictionary not already created
            units[splitline[0]] = [ splitline[2]]
            lessons[splitline[0]] = [ splitline[3]]
    return units, lessons




def main(is_sessions=False):
    read_file = os.path.expanduser('~/sorted_data/khan_data_sorted.csv')
    print(read_file)
    # [bylearnerorsession]
    if is_sessions:
        write_path = '~/cahl_output/summarize_stuckness_bysessions_problemtype.csv'
    else:
        write_path = '~/cahl_output/summarize_stuckness_bylearner_problemtype.csv'
    write_file = os.path.expanduser(write_path)
    stick = SummarizeStuckness(read_file, write_file)
    prerequisites = read_prerequisite_data('multilevel_prerequisites')
    units, lessons = read_topic_tree_data('math_topic_tree')
    stick.iterate_through_lines(prerequisites, units, lessons, sessions=is_sessions)

if __name__ == '__main__':
    start = time.time() 
    main()
    end =time.time()
    print(end-start)
    

