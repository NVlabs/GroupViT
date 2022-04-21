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
import sys
import zipfile

import numpy as np
import pandas as pd
import webdataset as wds
from tqdm import tqdm
import mmcv

def write_dataset(args):

    df = pd.read_csv(
        args.info, sep='\t', index_col='file', dtype=str, lineterminator='\n')
    print(f'Loaded dataframe: \n{df}')
    print(f'Length: \n{len(df)}')

    # This is the output pattern under which we write shards.
    pattern = os.path.join(args.shards, f'yfcc14m-%06d.tar')

    with wds.ShardWriter(
            pattern, maxsize=int(args.maxsize),
            maxcount=int(args.maxcount)) as sink:
        sink.verbose = 0
        all_keys = set()

        skipped = 0
        zip_files = list(mmcv.scandir(args.root, suffix='zip'))
        for idx, file in tqdm(
                enumerate(zip_files), desc='total', total=len(zip_files)):
            with zipfile.ZipFile(osp.join(args.root, file), 'r') as zfile:
                filename_list = zfile.namelist()
                for filename in tqdm(
                        filename_list, position=1, desc=f'{file}', leave=None):
                    image = zfile.read(filename)
                    if image is None:
                        skipped += 1
                        tqdm.write(f'Skipping {filename}, {skipped}/{len(df)}')
                        continue
                    fname = filename.replace('data/images/', '')
                    # Construct a unique key from the filename.
                    key = os.path.splitext(os.path.basename(fname))[0]

                    # Useful check.
                    if key in all_keys:
                        tqdm.write(f'duplicate: {fname}')
                        continue
                    assert key not in all_keys
                    all_keys.add(key)

                    text = str(df.loc[fname]['caption'])

                    if len(text.split(' ')) < 2:
                        skipped += 1
                        tqdm.write(f'Text {text} too short')
                        tqdm.write(f'Skipping {fname}, {skipped}/{len(df)}')
                        continue

                    # Construct a sample.
                    xkey = key
                    sample = {'__key__': xkey, 'jpg': image, 'text': text}

                    # Write the sample to the sharded tar archives.
                    sink.write(sample)
        print(f'skipped: {skipped}/{len(df)}')
        print(f'total keys: {len(all_keys)}')


def parse_args():
    parser = argparse.ArgumentParser(
        """Generate sharded dataset from original ImageNet data.""")
    parser.add_argument('--maxsize', type=float, default=1e9)
    parser.add_argument('--maxcount', type=float, default=100000)
    parser.add_argument('--shards', help='directory where shards are written')
    parser.add_argument('--root', help='data root path')
    parser.add_argument('--info', help='tsv path')
    args = parser.parse_args()

    assert args.maxsize > 10000000
    assert args.maxcount < 1000000
    return args


def main():
    args = parse_args()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if not os.path.isdir(os.path.join(args.shards, '.')):
        print(
            f'{args.shards}: should be a writable destination directory for shards',
            file=sys.stderr)
        sys.exit(1)

    write_dataset(args=args)


if __name__ == '__main__':
    main()
