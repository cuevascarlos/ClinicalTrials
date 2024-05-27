import os
import argparse
from scipy.stats import ranksums

'''
Script that computes statistical analysis of evaluation metrics between different runs of the same model
Author: Carlos Cuevas Villarmin
Last update: 27/05/2024
'''

def find_directories_with_name(directory, name):
    """
    Find directories within a given directory that have a specific name or start with a specific name.

    Args:
        directory (str): The directory to search within.
        name (str): The name or prefix of the directories to find.

    Returns:
        list: A list of directory paths that match the given name or prefix.
    """
    matching_directories = []
    for root, dirs, _ in os.walk(directory):
        for dir_name in dirs:
            if dir_name.startswith(name):
                reports_dir = os.path.join(root, dir_name, 'REPORTS')
                if os.path.exists(reports_dir):
                    matching_directories.append(reports_dir)
    return matching_directories

def find_files_with_name(directory, name):
    """
    Find files within a given directory that have a specific name or start with a specific name.

    Args:
        directory (str): The directory to search within.
        name (str): The name or prefix of the files to find.

    Returns:
        list: A list of file paths that match the given name or prefix.
    """
    matching_files = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.startswith(name):
                matching_files.append(os.path.join(root, file_name))
    return matching_files

def read_csv_files(file_name):
    """
    Read a csv file and return a list of dictionaries with the content of the file.

    Args:
        file_name (str): The name of the file to read.

    Returns:
        list: A list of dictionaries with the content of the file.
    """
    with open(file_name, 'r') as file:
        lines = file.readlines()
        headers = lines[0].strip().split(',')
        data_dict = {}
        for line in lines[1:]:
            values = line.strip().split(',')
            data_dict[values[0]] = []            
            for i, value in enumerate(values[2:]):
                if 'report' in headers[i+2]:
                    try:
                        data_dict[values[0]].append(float(value))
                    except:
                        pass
    return data_dict


if __name__ == '__main__':
    

    #Define the argument parser
    parser = argparse.ArgumentParser(description='Statistical analysis of evaluation metrics between different runs of the same model')
    parser.add_argument('-d', '--directory', help='Directory to search for files', required=True)
    parser.add_argument('-m', '--model', help='Model to compare', required=True)

    #Parse the arguments
    args = vars(parser.parse_args())
    directory = args['directory']
    model = args['model']

    directories = find_directories_with_name(directory, model)
    print(directories)
    data = {}
    for direct in directories:
        files = find_files_with_name(direct, 'f1_scores_strict_token_level')
        for file in files:
            data[file] = read_csv_files(file)

    f = open(f'./wilcoxon_test_{model}.txt', 'w')
    for i, elem in enumerate(data):
        for j in range(i+1,len(data)):
            against = list(data.items())[j][0]
            f.write(f"Wilcoxon test between {elem.split('/')[-4]} and {against.split('/')[-4]}: \n")
            print(f"Wilcoxon test between {elem.split('/')[-4]} and {against.split('/')[-4]}: \n")
            for key in data[elem].keys():
                #Wilcoxon test
                wilcoxon_result = ranksums(data[elem][key], data[against][key])
                f.write(f"\t Based on {key}: \t p-value = {round(wilcoxon_result.pvalue,4)}\n")
                print(f"\t Based on {key}: \t p-value = {round(wilcoxon_result.pvalue,4)}")
            print("\n")
            f.write("\n")
    f.close()
