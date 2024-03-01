import transformers
import pandas as pd
import re
import torch
import pathlib
import pickle
import pymorphy2
import os
import numpy as np
import itertools
import functools
import operator
from typing import Callable, Set, FrozenSet, Dict
from .datagen import apply_pattern
from .common import phase, ADJ_FEATS, PAST_VERB_FEATS, GENDERS

cache_dir = os.environ.get('TOKEN_CACHE')
if cache_dir is None:
    cache_dir = '.cache'
cache_dir = pathlib.Path(cache_dir)

def get_wordforms():
    outfile = cache_dir / "_wordforms.pkl"
    outfile.parent.mkdir(exist_ok=True, parents=True)
    try:
        with outfile.open('rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        pass

    with phase("No wordform cache available, building it..."):
        to_extract = [ADJ_FEATS, PAST_VERB_FEATS]
        by_gender = [[frozenset(spec | {gender}) for gender in GENDERS] for spec in to_extract]
        extracted = {grammemes:set() for grammemes in itertools.chain(*by_gender)}

        for parse in pymorphy2.MorphAnalyzer(lang='ru').iter_known_word_parses():
            for grammemes, wordforms in extracted.items():
                if grammemes in parse.tag:
                    wordforms.add(parse.word)
                    wordforms.add(parse.word.capitalize())

        for group in by_gender:
            intersection = functools.reduce(operator.and_, map(extracted.get, group))
            print(f"Intersection for group {group} is {len(intersection)} words long: {intersection}")
            for grammemes in group:
                extracted[grammemes] -= intersection

        with outfile.open('wb') as f:
            pickle.dump(extracted, f, pickle.HIGHEST_PROTOCOL)

        return extracted

def get_tokenized_wordforms(tokenizer: transformers.PreTrainedTokenizer):
    outfile = cache_dir / f"{tokenizer.name_or_path}.pkl"
    outfile.parent.mkdir(exist_ok=True, parents=True)
    try:
        with outfile.open('rb') as f:
            return pickle.load(f)
    except:
        pass

    wordforms = get_wordforms()

    with phase(f"No token cache available for model {tokenizer.name_or_path}, building it..."):
        tokenized = {
            grammemes: tokenizer(list(wordform_set), add_special_tokens=False)["input_ids"]
            for grammemes, wordform_set
            in wordforms.items()
        }

        with outfile.open('wb') as f:
            pickle.dump(tokenized, f, pickle.HIGHEST_PROTOCOL)

        return tokenized

def make_subword_patterns(patterns: pd.DataFrame, tokenizer: transformers.PreTrainedTokenizer) -> pd.DataFrame:
    assert patterns.columns[0] == 'pattern'
    variants = patterns.columns[1:]
    def find_endings(row):
        prefix = None
        endings = []
        for variant in row[variants]:
            tokens = tokenizer.encode(variant, add_special_tokens=False)
            if len(tokens) == 1:
                print(f"Skipping pattern '{row['pattern']} because '{variant}' is a single token")
                return {variant: pd.NA for variant in variants}
            if prefix is None:
                prefix = tokens[:-1]
            else:
                if prefix != tokens[:-1]:
                    print(f"Skipping pattern '{row['pattern']}' due to mismatched tokenization")
                    return {variant: pd.NA for variant in variants}
            endings.append(tokens[-1])
        ret = {variant_name: ending for variant_name, ending in zip(variants, endings)}
        prefix = tokenizer.decode(prefix)
        ret['pattern'], nsubs = re.subn(r"(?=\{mask\})", prefix, row['pattern'])
        assert nsubs == 1
        return ret
    return patterns.apply(find_endings, axis='columns', result_type='expand').dropna()

def mask(base: pd.DataFrame, pattern: str, tokenizer: transformers.PreTrainedTokenizer, mask_length=1):
    dataset = apply_pattern(base, pattern, "".join(itertools.repeat(tokenizer.mask_token, mask_length)))
    tokenized = tokenizer(dataset["sent"].to_list(), return_offsets_mapping=True, padding=True)
    for key, value in tokenized.items():
        dataset[key] = value
    dataset['mask_locations'] = dataset['input_ids'].map(lambda l: [token == tokenizer.mask_token_id for token in l])
    return dataset

def make_simple_probs_map(target_tokens: pd.Series) -> Callable[[torch.Tensor],pd.DataFrame]:
    def f(probs: torch.Tensor) -> pd.DataFrame:
        ret = target_tokens.apply(lambda idx: pd.Series(probs[:,idx].numpy()))
        return ret.T
    return f

def make_whole_token_probs_map(tokenizer: transformers.PreTrainedTokenizer, grammeme_map: Dict[str, FrozenSet[str]], mask_length=1):
    tokens = get_tokenized_wordforms(tokenizer)
    idxs = {tag: torch.tensor([wordform
             for wordform
             in tokens[grammemes]
             if len(wordform) == mask_length
    ]) for tag, grammemes in grammeme_map.items()}
    def f(probs: torch.Tensor) -> pd.DataFrame:
        per_sent = probs.reshape(-1, mask_length, probs.shape[-1])
        ret = {}
        for tag, to_take in idxs.items():
            to_stack = [per_sent[:,idx,idx_tokens] for idx, idx_tokens in enumerate(to_take.T)]
            prob_sums = torch.stack(to_stack, dim=-1).prod(-1).sum(-1)
            ret[tag] = prob_sums.tolist()
        return pd.DataFrame.from_dict(ret)
    return f