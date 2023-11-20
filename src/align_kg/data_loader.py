import string
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from datasets import load_dataset, concatenate_datasets, Dataset
from utils import rule_to_string


def load_new_tokens(default_new_tokens, rel_dict_path):
    if isinstance(rel_dict_path, str):
        rel_dict_path = [rel_dict_path]
    for rel_path in rel_dict_path:
        with open(rel_path, 'r') as f:
            for line in f.readlines():
                _, r = line.strip().split('\t')
                default_new_tokens.append(r)
    return default_new_tokens
        

def load_multiple_datasets(data_path_list, shuffle=False):
    '''
    Load multiple datasets from different paths.

    Args:
        data_path_list (_type_): _description_
        shuffle (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    '''
    dataset_list = [load_dataset('json', data_files=p, split="train")
                     for p in data_path_list]
    dataset = concatenate_datasets(dataset_list)
    if shuffle:
        dataset = dataset.shuffle()
    return dataset



def get_test_dataset(dataset):
    # Gather all the labels for the same question
    test_dataset = dict()
    for sample in dataset:
        if sample['question'] not in test_dataset:
            test_dataset[sample['question']] = set()
        label = tuple(sample['path'])
        test_dataset[sample['question']].add(label)
    test_dataset = [{'text': k, 'label': v} for k, v in test_dataset.items()]
    return Dataset.from_list(test_dataset)


