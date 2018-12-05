import time
import csv
import os
import pdb

'''
Script to iterate through the script
'''

class SummarizeStuckness():

    def __init__(self, read_filename, write_filename):
        print('initialize '+ read_filename)
        self.reader = open(read_filename,'r')
        self.writefile = open(write_filename, 'w')
       
    def iterate_through_lines(self):
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
        first_line = self.reader.readline()
        counter = 1
        for line in self.reader:
            self.parse_line(line)
            counter+=1
            if counter % 1000000 == 0:
                print(counter)
    
    def write_header(self):
        self.csvwriter.writerow([ 'sha_id', 
            'total_problems',
            'never_stuck_problems', 
            'never_unstuck_problems',
            'unstuck_problems',
            'unstuck_same_exercise',
            'unstuck_remediation_problems',
            'unstuck_correct_remdiation_problems'])
    
    def parse_line(self, line):
        '''
           Parse through each line and store the values 
        '''
        line_delimited = line.split(',') 
        sha_id = line_delimited[0]
        exercise = line_delimited[5]
        problem_type = line_delimited[7]
        correct = line_delimited[8] == 'true'
        attempt_numbers = int(line_delimited[12])
        problem = exercise + '|' +  problem_type
        if sha_id != self.last_sha_id:
            self.summarize_old_sha_id()
            self.last_sha_id = sha_id
            self.last_problem = problem
            self.user_attempts = {}
            self.update_attempts(correct, attempt_numbers, problem) 
        else:
            self.update_attempts(correct, attempt_numbers, problem)  
            self.add_new_data_for_user( problem_type, exercise, correct)
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
             

    def summarize_old_sha_id(self):
        '''
            summarize user data 
            and write the array of summary stats
            to the file
        '''
        never_stuck_problems = len(self.user_data['never_stuck'])
        never_unstuck_problems = len(self.user_data['stuck'].keys())
        unstuck_problems = len(self.user_data['unstuck'].keys())
        unstuck_same_exercise = 0
        # [TODO: add] unstuck_diff_exercise_same_unit = 0
        # [TODO] unstuck_has_prereq_problems = 0
        # [TODO] unstuck_prereq_avail_problems = 0
        unstuck_remediation_problems = 0 
        unstuck_correct_remdiation_problems = 0
        for unstuck_item in self.user_data['unstuck']:
            unstuck_array = self.user_data['unstuck'][unstuck_item] 
            unstuck_same_exercise += unstuck_array['same_exercise_remediation_problems']
            unstuck_remediation_problems += unstuck_array['remediation_problems']
            unstuck_correct_remdiation_problems += unstuck_array['correct_remediation_problems']
        self.csvwriter.writerow([ self.last_sha_id, 
            never_unstuck_problems + never_stuck_problems +  unstuck_problems,
            never_stuck_problems, 
            never_unstuck_problems,
            unstuck_problems,
            unstuck_same_exercise,
            unstuck_remediation_problems,
            unstuck_correct_remdiation_problems])
        # clear user data 
        self.user_data = {'stuck':{},'unstuck':{}, 'never_stuck':[], 'stuck_correct':{}  }

    def add_new_data_for_user(self, problem_type, exercise, correct):
        problem = exercise + '|' + problem_type 
        if self.user_attempts[problem]['correct']>=2 and problem in self.user_data['stuck']:
            self.user_data['unstuck'][problem]  = self.summarize_unstuck(problem)
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
    
    def summarize_unstuck(self, problem):
        '''
            output: output the unstuck summary stats array
            which lists the attibutes of the unstuckness token
            ['same_exercise','remediation_problems','correct_remediation_problems']
        '''
        # [TODO] add the is same unit and is a prerequsitie count
        exercise = problem.split('|')[0]
        correct_list = self.user_data['stuck_correct'][problem]
        remediation_list = self.user_data['stuck'][problem]
        same_exercise_list = [item.split('|')[0] == exercise for item in remediation_list]
        # [TODO] if the current problem is equal to the stuck problem name
        # then add to stuck metric
        unstuck_state = {} 
        unstuck_state['same_exercise_remediation_problems'] = sum(same_exercise_list) 
        unstuck_state['remediation_problems'] = len(remediation_list) 
        unstuck_state['correct_remediation_problems'] = sum(correct_list)
        return unstuck_state



def main():
    read_file = os.path.expanduser('~/sorted_data/khan_data_sorted.csv')
    print(read_file)
    write_file = os.path.expanduser('~/cahl_output/summarize_stuckness_bylearner_problemtype.csv')
    stick = SummarizeStuckness(read_file, write_file)
    stick.iterate_through_lines()

if __name__ == '__main__':
    start = time.time() 
    main()
    end =time.time()
    print(end-start)
    

