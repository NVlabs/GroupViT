# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

import os
import os.path as osp
import argparse

import pandas as pd
import sqlite3
import pandas as pd
import os.path as osp
from urllib.parse import unquote
import re
from datadings.tools import locate_files
from yfcc100m.vars import FILES
from yfcc100m.convert_metadata import download_db
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)


def key2path(key):
    img_path = osp.join(key[0:3], key[3:6], key + '.jpg')
    return img_path


def clean_caption(line):
    line = unquote(str(line))
    line = remove_html_tags(line)
    return line.replace('\n', ' ').replace('+', ' ')


def remove_html_tags(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def parse_args():
    parser = argparse.ArgumentParser(description='Create YFCC subset sql db and tsv')
    parser.add_argument('--input-dir', help='input sql db file directory')
    parser.add_argument('--output-dir', help='output tsv directory')
    parser.add_argument(
        '--subset', help='subset of data to use', default='yfcc100m_subset_data.tsv')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    files = locate_files(FILES, args.input_dir)
    # download DB file with AWS tools
    download_db(files)

    fullset_name = 'yfcc100m_dataset'
    subset_name = 'yfcc14m_dataset'
    conn = sqlite3.connect(osp.join(args.input_dir, 'yfcc100m_dataset.sql'))
    # get column names
    # some settings that hopefully speed up the queries
    # conn.execute(f'PRAGMA query_only = YES')
    conn.execute(f'PRAGMA journal_mode = OFF')
    conn.execute(f'PRAGMA locking_mode = EXCLUSIVE')
    conn.execute(f'PRAGMA page_size = 4096')
    conn.execute(f'PRAGMA mmap_size = {4*1024*1024}')
    conn.execute(f'PRAGMA cache_size = 10000')

    print('reading subset data')
    subset_df = pd.read_csv(args.subset, sep='\t', usecols=[1, 2], names=['photoid', 'photo_hash'], index_col='photoid')
    subset_df.to_sql(subset_name, con=conn, if_exists='replace')

    print('overwriting with subset')
    select_query = f'select {fullset_name}.*, {subset_name}.photo_hash from {fullset_name} inner join {subset_name} on {fullset_name}.photoid = {subset_name}.photoid'
    new_name = 'yfcc100m_dataset_new'
    print('creating new table')
    conn.execute(f'drop table if exists {new_name}')
    conn.execute(' '.join([f'create table {new_name} as ', select_query]))
    print(f'droping {fullset_name}')
    conn.execute(f'drop table if exists {fullset_name}')
    print(f'droping {subset_name}')
    conn.execute(f'drop table if exists {subset_name}')
    print(f'renaming {new_name} to {fullset_name}')
    conn.execute(f'alter table {new_name} rename to {fullset_name}')
    print('vacuuming db')
    conn.execute('vacuum')

    print(f'Loading dataframe from SQL')
    anno_df = pd.read_sql(f'select * from {fullset_name}', con=conn)
    print(f'Loaded dataframe from SQL: \n{anno_df.head()}')
    print(f'Length: \n{len(anno_df)}')
    print(f'generating filepath')
    anno_df['file'] = anno_df['photo_hash'].parallel_map(key2path)
    anno_df['caption'] = anno_df['description'].parallel_map(clean_caption)
    anno_df = anno_df[['file', 'caption']]
    print(f'Generated dataframe: \n{anno_df.head()}')

    print('saving subset as tsv')
    os.makedirs(args.output_dir, exist_ok=True)
    anno_df.to_csv(osp.join(args.output_dir, 'yfcc14m_dataset.tsv'), sep='\t', index=False)
    conn.close()


if __name__ == '__main__':
    main()
