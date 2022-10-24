import argparse
import json
import os
import os.path as osp
from collections import defaultdict

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='comare results')
    parser.add_argument('--files',
                        help='input json file in format of type:file_path,'
                        ' e.g.: FP32:xxx.json',
                        nargs='+')
    parser.add_argument('--output', default='output.xlsx')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    data = defaultdict(list)
    for pair in args.files:
        type, file_path = pair.strip().split(':')
        if not osp.exists(file_path):
            print(f'File not exists: {file_path}')
            continue
        data['type'].append(type)
        with open(file_path, 'r') as f:
            content = json.load(f)
            for k, v in content.items():
                data[k].append(v)
    df = pd.DataFrame(data)
    output_dir, _ = osp.split(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_excel(args.output)
    print(df.T)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
