import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import argparse
import os
import json
from datasets import load_dataset
import multiprocessing as mp
import utils
from tqdm import tqdm
from functools import partial

def build_data(args):
    '''
    Extract the paths between question and answer entities from the dataset.
    '''
    
    input_file = os.path.join(args.data_path, args.d)
    output_dir = os.path.join(args.output_path, args.d)
    
    print("Save results to: ", output_dir)
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    
    
    # Load dataset
    dataset = load_dataset(input_file, split=args.split)
    with open(os.path.join(output_dir, args.save_name), 'w') as fout:
        with mp.Pool(args.n) as pool:
                
                for res in tqdm(pool.imap_unordered(partial(process_data, remove_duplicate=args.remove_duplicate), dataset), total=len(dataset)):
                    for r in res:
                        fout.write(json.dumps(r) + '\n')

def process_data(data, remove_duplicate=False):
    question = data['question']
    graph  =  utils.build_graph(data['graph'])
    paths = utils.get_truth_paths(data['q_entity'], data['a_entity'], graph)
    result = []
    # Split each Q-P pair into a single data
    rel_paths = []
    for path in paths:
        rel_path = [p[1] for p in path] # extract relation path
        if remove_duplicate:
            if tuple(rel_path) in rel_paths:
                continue
        rel_paths.append(tuple(rel_path))
    for rel_path in rel_paths:
        result.append({"question": question, "path": rel_path})
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="rmanluo")
    parser.add_argument('--d', '-d', type=str, default='webqsp')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument("--output_path", type=str, default="datasets/AlignData")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument('--n', '-n', type=int, default=1)
    parser.add_argument('--remove_duplicate', action='store_true')
    args = parser.parse_args()
    
    if args.save_name == "":
        args.save_name = args.d + "_" + args.split + ".jsonl"
    
    build_data(args)
