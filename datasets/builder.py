# -------------------------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE
#
# Written by Ze Liu, Zhenda Xie
# Modified by Jiarui Xu
# -------------------------------------------------------------------------

import os.path as osp
import random
import warnings
from functools import partial

import nltk
import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds
from braceexpand import braceexpand
from mmcv.parallel import collate
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import _pil_interp
from torchvision import transforms

from .formatting import ToDataContainer
from .tokenizer import SimpleTokenizer


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(config):
    local_rank = dist.get_rank() % torch.cuda.device_count() if dist.is_initialized() else 0

    dataset_train = build_dataset(is_train=True, config=config)
    print(f'local rank {local_rank} / global rank {dist.get_rank()} \
        successfully build train dataset')
    dataset_val = build_dataset(is_train=False, config=config)
    print(f'local rank {local_rank} / global rank {dist.get_rank()} \
        successfully build val dataset')

    dc_collate = partial(collate, samples_per_gpu=config.batch_size)
    train_len = len(dataset_train)
    init_fn = partial(worker_init_fn, num_workers=config.num_workers, rank=dist.get_rank(), seed=config.seed)
    data_loader_train = wds.WebLoader(
        dataset_train.batched(config.batch_size, dc_collate, partial=False),
        batch_size=None,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=init_fn)

    train_nbatches = max(1, train_len // (config.batch_size * dist.get_world_size()))
    data_loader_train = (data_loader_train.with_epoch(train_nbatches).with_length(train_nbatches))

    data_loader_val = wds.WebLoader(
        dataset_val.batched(config.batch_size, dc_collate),
        batch_size=None,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=init_fn)

    val_len = len(dataset_val)
    val_nbatches = max(1, val_len // (config.batch_size * dist.get_world_size()))
    data_loader_val = (data_loader_val.with_epoch(val_nbatches).with_length(val_nbatches))

    return dataset_train, dataset_val, data_loader_train, data_loader_val


def warn_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning,
    and continue."""
    warnings.warn(repr(exn))
    return True


def build_dataset(is_train, config):
    img_transform = build_img_transform(is_train, config.img_aug)
    text_transform = build_text_transform(is_train, config.text_aug)
    split = 'train' if is_train else 'val'
    dataset_type = None
    tar_file_list = []
    total_length = 0
    for ds in config.dataset[split]:
        ds_meta = config.dataset.meta[ds]
        if dataset_type is None:
            dataset_type = ds_meta.type
        else:
            assert dataset_type == ds_meta.type, \
                'All datasets must be of the same type'

        prefix = ds_meta.prefix
        path = ds_meta.path
        length = ds_meta.length
        cur_tar_file_list = []
        for tar_file in braceexpand(osp.join(path, prefix)):
            if osp.exists(tar_file):
                cur_tar_file_list.append(tar_file)
        print(f'Found {len(cur_tar_file_list)} files for dataset {ds}')
        tar_file_list.extend(cur_tar_file_list)
        total_length += length
    print(f'Found {len(tar_file_list)} files in total for split {split}')
    # yapf: disable
    if is_train:
        dataset = (  # noqa
            wds.WebDataset(tar_file_list, repeat=True, handler=warn_and_continue)
            .shuffle(config.shuffle_buffer)
            .decode('pil', handler=warn_and_continue)
            .rename(image='jpg;png;jpeg', text='text;txt', keep=False, handler=warn_and_continue)
            .map_dict(image=img_transform, text=text_transform, handler=warn_and_continue)
            .with_length(total_length))
    else:
        # zero shot classification validation
        dataset = (  # noqa
            wds.WebDataset(tar_file_list, repeat=False, handler=warn_and_continue)
            .shuffle(0)
            .decode('pil', handler=warn_and_continue)
            .rename(image='jpg;png;jpeg', target='cls', keep=False)
            .map_dict(image=img_transform, target=ToDataContainer())
            .slice(dist.get_rank(), total_length, dist.get_world_size())
            .with_length(total_length))
    # yapf: enable

    return dataset


def build_img_transform(is_train, config, with_dc=True):

    if not config.deit_aug:
        if is_train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(config.img_size, scale=config.img_scale),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(config.img_size + 32),
                transforms.CenterCrop(config.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ])

        return transform

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.img_size,
            is_training=True,
            color_jitter=config.color_jitter if config.color_jitter > 0 else None,
            auto_augment=config.auto_augment if config.auto_augment != 'none' else None,
            re_prob=config.re_prob,
            re_mode=config.re_mode,
            re_count=config.re_count,
            interpolation=config.interpolation,
        )
    else:
        size = int((256 / 224) * config.img_size)
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=_pil_interp(config.interpolation)),
            transforms.CenterCrop(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])

    if with_dc:
        transform = transforms.Compose([*transform.transforms, ToDataContainer()])

    return transform


def build_text_transform(is_train, config, with_dc=True):
    local_rank = dist.get_rank() % torch.cuda.device_count() if dist.is_initialized() else 0
    if config.multi_label and is_train:
        # only down on local rank 0
        if local_rank == 0:
            nltk.download('popular')
        transform = WordAugTokenizeWrapper(
            Tokenize(SimpleTokenizer(), max_seq_len=config.max_seq_len),
            max_word=config.multi_label,
            word_type=config.word_type)

    else:
        transform = Tokenize(SimpleTokenizer(), max_seq_len=config.max_seq_len)

    if with_dc:
        transform = transforms.Compose([transform, ToDataContainer()])

    return transform


class Tokenize:

    def __init__(self, tokenizer, max_seq_len=77, truncate=True):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.truncate = truncate

    def __call__(self, texts):
        expanded_dim = False
        if isinstance(texts, str):
            texts = [texts]
            expanded_dim = True

        sot_token = self.tokenizer.encoder['<|startoftext|>']
        eot_token = self.tokenizer.encoder['<|endoftext|>']
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), self.max_seq_len, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.max_seq_len:
                if self.truncate:
                    tokens = tokens[:self.max_seq_len]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f'Input {texts[i]} is too long for context length {self.max_seq_len}')
            result[i, :len(tokens)] = torch.tensor(tokens)

        if expanded_dim:
            return result[0]

        return result


class WordAugTokenizeWrapper:

    def __init__(self, tokenize, max_word=3, template_set='full', word_type='noun'):
        self.tokenize = tokenize
        self.max_word = max_word
        from .imagenet_template import (full_imagenet_templates, sub_imagenet_template, simple_imagenet_template,
                                        identity_template)
        assert template_set in ['full', 'subset', 'simple', 'identity']
        if template_set == 'full':
            templates = full_imagenet_templates
        elif template_set == 'subset':
            templates = sub_imagenet_template
        elif template_set == 'simple':
            templates = simple_imagenet_template
        elif template_set == 'identity':
            templates = identity_template
        else:
            raise ValueError
        self.templates = templates
        assert word_type in ['noun', 'noun_phrase']
        self.word_type = word_type

    def get_tag(self, tokenized, tags):
        if not isinstance(tags, (list, tuple)):
            tags = [tags]
        ret = []
        for (word, pos) in nltk.pos_tag(tokenized):
            for tag in tags:
                if pos == tag:
                    ret.append(word)
        return ret

    def get_noun_phrase(self, tokenized):
        # Taken from Su Nam Kim Paper...
        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR>}
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        """
        chunker = nltk.RegexpParser(grammar)

        chunked = chunker.parse(nltk.pos_tag(tokenized))
        continuous_chunk = []
        current_chunk = []

        for subtree in chunked:
            if isinstance(subtree, nltk.Tree):
                current_chunk.append(' '.join([token for token, pos in subtree.leaves()]))
            elif current_chunk:
                named_entity = ' '.join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        return continuous_chunk

    def __call__(self, text):
        assert isinstance(text, str)
        tokenized = nltk.word_tokenize(text)
        nouns = []
        if len(tokenized) > 0:
            if self.word_type == 'noun':
                nouns = self.get_tag(tokenized, ['NN', 'NNS', 'NNP', 'VBG', 'VB', 'VBD', 'VBN', 'VBP', 'VBZ'])
            elif self.word_type == 'noun_phrase':
                nouns = self.get_noun_phrase(tokenized)
            else:
                raise ValueError('word_type must be noun or noun_phrase')

        prompt_texts = []
        if len(nouns) > 0:
            select_nouns = np.random.choice(nouns, min(self.max_word, len(nouns)), replace=False)
            prompt_texts = [np.random.choice(self.templates).format(noun) for noun in select_nouns]
        if len(prompt_texts) < self.max_word:
            prompt_texts += [text] * (self.max_word - len(prompt_texts))

        texts = [text] + prompt_texts
        return self.tokenize(texts)
