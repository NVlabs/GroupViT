# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

import argparse
import json
import os
import os.path as osp
import random

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input', type=str, help='path to redcaps annotations directory')
    parser.add_argument(
        'output', type=str, help='output annotations file path')
    parser.add_argument(
        '--num-split', type=int, help='number of splits to make')
    return parser


def main(args):
    annos = []
    for fname in tqdm.tqdm(os.listdir(args.input), desc='merging json files'):
        if fname.endswith('json'):
            with open(os.path.join(args.input, fname)) as f:
                a = json.load(f)
                for d in a['annotations']:
                    cur_d = {'URL': d['url'], 'TEXT': d['caption']}
                    annos.append(cur_d)

    random.seed(42)
    random.shuffle(annos)
    if args.num_split is None:
        df = pd.DataFrame(annos)
        print(df.head())
        print(f'saving {len(df)} annotations to {args.output}')
        table = pa.Table.from_pandas(df)
        os.makedirs(osp.dirname(args.output), exist_ok=True)
        pq.write_table(table, args.output)
    else:
        for i in range(args.num_split):
            df = pd.DataFrame(annos[i::args.num_split])
            print(df.head())
            output = osp.splitext(
                args.output)[0] + f'_part{i}{osp.splitext(args.output)[1]}'
            print(f'saving {len(df)} annotations to {output}')
            table = pa.Table.from_pandas(df)
            os.makedirs(osp.dirname(output), exist_ok=True)
            pq.write_table(table, output)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
