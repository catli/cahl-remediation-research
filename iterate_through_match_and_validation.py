import getopt, sys
import pdb
from create_similar_token_sets import main as create_similar_token
from validate_learning_similarity import main as validate_learning_similarity


fullCmdArguments = sys.argv
argumentList = fullCmdArguments[1:]
unixOptions = 'r:m:c:s:'
gnuOptions = ['readfileaffix=','method=','comparison=','sample=']

try:
    arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
except:
    sys.exit(2)


# def pass_in_arguments():
#     for currentArgument, currentValue in arguments:
#         if currentArgument in ['-r','--readfileaffix']:
#             read_file_affix = currentValue
#             print( ("read file affix with: %s") % (read_file_affix))
#         if currentArgument in '--method':
#             method = currentValue
#             print( ("read file affix with: %s") % (method))
#         if currentArgument in '--comparison':
#             find_nearest_comparison = currentValue
#             print( ("with comparison: %s") % (find_nearest_comparison))
#         if currentArgument in '--sample':
#             remediation_sample_number = int(currentValue)
#             print( ("with sample: %s") % (remediation_sample_number))

#         create_similar_token(
#                     read_file_affix = read_file_affix,
#                     method = method,
#                     find_nearest_comparison = find_nearest_comparison,
#                     remediation_sample_number = remediation_sample_number )

#         validate_learning_similarity(
#                     read_file_affix = read_file_affix,
#                     method = method,
#                     find_nearest_comparison = find_nearest_comparison,
#                     remediation_sample_number = remediation_sample_number)





parameters = {
    "responseCosine1": {"read_file_affix": 'full', "method": 'cosine', 
        "find_nearest_comparison": 'response', "remediation_sample_number": 1},
    "responseCosine5": {"read_file_affix": 'full', "method": 'cosine', 
        "find_nearest_comparison": 'response', "remediation_sample_number": 5},
    "responseCosine10": {"read_file_affix": 'full', "method": 'cosine', 
        "find_nearest_comparison": 'response', "remediation_sample_number": 10},
    "responseCosine20": {"read_file_affix": 'full', "method": 'cosine', 
        "find_nearest_comparison": 'response', "remediation_sample_number": 20},
    "learnCosine1": {"read_file_affix": 'full', "method": 'cosine', 
        "find_nearest_comparison": 'learn', "remediation_sample_number": 1},
    "learnCosine5": {"read_file_affix": 'full', "method": 'cosine', 
        "find_nearest_comparison": 'learn', "remediation_sample_number": 5},
    "learnCosine10": {"read_file_affix": 'full', "method": 'cosine', 
        "find_nearest_comparison": 'learn', "remediation_sample_number": 10},
    "learnCosine20": {"read_file_affix": 'full', "method": 'cosine', 
        "find_nearest_comparison": 'learn', "remediation_sample_number": 20},
    "responseEuclidean1": {"read_file_affix": 'full', "method": 'euclidean', 
        "find_nearest_comparison": 'response', "remediation_sample_number": 1},
    "responseEuclidean5": {"read_file_affix": 'full', "method": 'euclidean', 
        "find_nearest_comparison": 'response', "remediation_sample_number": 5},
    "responseEuclidean10": {"read_file_affix": 'full', "method": 'euclidean', 
        "find_nearest_comparison": 'response', "remediation_sample_number": 10},
    "exerciseCosine10": {"read_file_affix": 'exercise', "method": 'cosine', 
        "find_nearest_comparison": 'response', "remediation_sample_number": 10},
    "exerciseCosine5": {"read_file_affix": 'exercise', "method": 'cosine', 
        "find_nearest_comparison": 'response', "remediation_sample_number": 5}
    # "exerciseCosine1": {"read_file_affix": 'exercise', "method": 'cosine', 
    #     "find_nearest_comparison": 'response', "remediation_sample_number": 1}
}

for iter in parameters:
    read_file_affix =  parameters[iter]["read_file_affix"]
    method =  parameters[iter]["method"]
    find_nearest_comparison =  parameters[iter]["find_nearest_comparison"]
    remediation_sample_number =  parameters[iter]["remediation_sample_number"]

    create_similar_token(
                read_file_affix = read_file_affix,
                method = method,
                find_nearest_comparison = find_nearest_comparison,
                remediation_sample_number = remediation_sample_number )

    validate_learning_similarity(
                read_file_affix = read_file_affix,
                method = method,
                find_nearest_comparison = find_nearest_comparison,
                remediation_sample_number = remediation_sample_number)