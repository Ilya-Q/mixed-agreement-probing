import argparse
import pathlib
import csv
import transformers
import pymorphy2
import numpy as np
import torch
import itertools
import functools
import os
import operator
import pickle
from time import time
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Dict, Set, FrozenSet

device='cpu'

ADJ_FEATS, PAST_VERB_FEATS = {"ADJF", "nomn"}, {'VERB', 'past', 'sing'}
GENDERS = ('femn', 'masc')

class phase:
    def __init__(self, message: str):
        self._message = message
    def __enter__(self):
        print(self._message)
        self._start = time()
        return self
    def __exit__(self, type, value, traceback):
        elapsed = time() - self._start
        print(f"Done in {elapsed:.4f} seconds")

def build_token_mask(tokenizer: transformers.PreTrainedTokenizer, grammemes: FrozenSet[str]):
    mask = np.zeros(len(tokenizer.get_vocab()), dtype=bool)
    for token_id in tokenizer.get_vocab().values():
        word = tokenizer.decode([token_id])
        word = word.strip().lstrip('_ ')
        if not word.startswith("##") and morph.word_is_known(word) and any(grammemes in parse.tag for parse in morph.parse(word)):
            mask[token_id] = True
    return mask

def get_wordforms():
    outfile = args.cache_dir / "_wordforms.pkl"
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

        for group in by_gender:
            intersection = functools.reduce(operator.and_, map(extracted.get, group))
            print(f"Intersection for group {group} is {len(intersection)} words long: {intersection}")
            for grammemes in group:
                extracted[grammemes] -= intersection

        with outfile.open('wb') as f:
            pickle.dump(extracted, f, pickle.HIGHEST_PROTOCOL)

        return extracted

def get_tokenized_wordforms(tokenizer: transformers.PreTrainedTokenizer):
    outfile = args.cache_dir / f"{tokenizer.name_or_path}.pkl"
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

def get_idx_for_grammemes(tokenizer: transformers.PreTrainedTokenizer, grammemes: Set[str]):
    grammemes = frozenset(grammemes)
    cache_path = pathlib.Path(args.cache_dir, tokenizer.name_or_path + '.pkl')
    try:
        with cache_path.open() as f:
            cache = pickle.load(f)
    except FileNotFoundError:
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        cache = {}

    try:
        return cache[grammemes]
    except KeyError:
        cache[grammemes] = build_token_mask(tokenizer, grammemes)
        with cache_path.open('w') as f:
            pickle.dump(f)
        return cache[grammemes]
    
def check_masked_targets_multi(
        pattern: str,
        nouns: List[str],
        fixed_targets: List[str],
        target_grammemes: Set[str],
        mask_grammemes: Set[str],
        max_len: int,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        out_file = pathlib.Path
):
    model.eval()
    fixed_targets = [next(p for p in morph.parse(t) if target_grammemes in p.tag) for t in fixed_targets]

    tokenized_wordforms = get_tokenized_wordforms(tokenizer)
    gender_idxs: dict[str, List[np.array]] = {}

    with phase("Getting indices for gendered tokens..."):
        for gender in GENDERS:
            grammemes_with_gender = frozenset(mask_grammemes | {gender})
            idxs = [[] for _ in range(max_len)]
            for tokens in tokenized_wordforms[grammemes_with_gender]:
                if len(tokens) <= max_len:
                    idxs[len(tokens)-1].append(tokens)
            gender_idxs[gender] = list(map(torch.tensor, idxs))

    out: List[Dict] = []
    sents: List[List[str]] = [[] for _ in range(max_len)]
    fields = ['noun', 'fixed_target', 'fixed_target_gender', 'femn_prob_sum', 'masc_prob_sum']
    fields.extend(f"{gender}_prob_sum_{length}_tok" for gender, length in itertools.product(GENDERS, range(1, max_len+1)))
    
    grammemes_for_gender = {gender: target_grammemes | {gender} for gender in GENDERS}
    with phase(f"Building test sentences..."):
        for noun, fixed, fixed_gender in itertools.product(nouns, fixed_targets, GENDERS):
            row = {"noun": noun, "fixed_target": fixed.word, "fixed_target_gender": fixed_gender}
            infl = fixed.inflect(grammemes_for_gender[fixed_gender])
            for i in range(max_len):
                sents[i].append(pattern.format(
                    noun=noun,
                    mask="".join(itertools.repeat(tokenizer.mask_token, i+1)),
                    fixed_target=infl.word
                ))
            out.append(row)

    with phase("Tokenizing..."):
        tokenized = [tokenizer(sents, return_tensors='pt', padding=True).to(device) for sents in sents]
        mask_locations = [tokenized['input_ids'] == tokenizer.mask_token_id for tokenized in tokenized]

    with phase(f"Sending model to device {device}..."):
        model.to(device)

    with phase("Running model..."):
        with torch.no_grad():
            preds = [model(**tokenized) for tokenized in tokenized]

    with phase("Computing probabilities..."):
        probs = [preds.logits.detach()[mask_locations].softmax(dim=1) for preds, mask_locations in zip(preds, mask_locations)]

    prob_sums = {gender:[] for gender in GENDERS}
    for length, probs in enumerate(probs, start=1):
        with phase(f"Summing probabilities for targets of length {length}..."):
            per_sent = probs.reshape(-1, length, probs.shape[-1])
            for gender in GENDERS:
                to_take = gender_idxs[gender][length-1]
                prob_sums[gender].append(
                    torch.stack([per_sent[:,idx,idx_tokens] for idx, idx_tokens in enumerate(to_take.T)], dim=-1).prod(-1).sum(-1)
                )

    prob_sums_total = {gender:sum(prob_sums[gender]) for gender in GENDERS}

    with phase("Saving probability sums..."):
        for i, row in enumerate(out):
            for gender in GENDERS:
                row[f"{gender}_prob_sum"] = prob_sums_total[gender][i].item()
                for length in range(1,max_len+1):
                    row[f"{gender}_prob_sum_{length}_tok"] = prob_sums[gender][length-1][i].item()
    
    with phase("Writing output file..."):
        with open(out_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(out)

    
def check_masked_targets(
        pattern: str,
        nouns: List[str],
        fixed_targets: List[str],
        target_grammemes: Set[str],
        mask_grammemes: Set[str],
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        out_file = pathlib.Path
):
    model.eval()
    fixed_targets = [next(p for p in morph.parse(t) if target_grammemes in p.tag) for t in fixed_targets]

    gender_masks: Dict[str, np.array] = {}
    for gender in GENDERS:
        with phase(f"Building mask for {gender}..."):
            grammemes_with_gender = mask_grammemes | {gender}
            gender_masks[gender] = build_token_mask(tokenizer, grammemes_with_gender)
            print(f"{gender_masks[gender].sum()} non-zero elements")

    with phase("Intersecting gender masks..."):
        intersection = functools.reduce(np.logical_and, gender_masks.values())
        print(intersection.sum())
        print(*(tokenizer.decode([token_id]) for token_id in np.flatnonzero(intersection)), sep=', ')
        for mask in gender_masks.values():
            mask ^= intersection

    out: List[Dict] = []
    sents: List[str] = []
    fields = ['noun', 'fixed_target', 'fixed_target_gender', 'femn_prob_sum', 'masc_prob_sum']

    grammemes_for_gender = {gender: target_grammemes | {gender} for gender in GENDERS}
    with phase(f"Building test sentences..."):
        for noun, fixed, fixed_gender in itertools.product(nouns, fixed_targets, GENDERS):
            row = {"noun": noun, "fixed_target": fixed.word, "fixed_target_gender": fixed_gender}
            infl = fixed.inflect(grammemes_for_gender[fixed_gender])
            sents.append(pattern.format(noun=noun, mask=tokenizer.mask_token, fixed_target=infl.word))
            out.append(row)

    with phase("Tokenizing..."):
        tokenized = tokenizer(sents, return_tensors='pt', padding=True).to(device)
    with phase("Finding mask tokens..."):
        mask_locations = tokenized['input_ids'] == tokenizer.mask_token_id

    with phase(f"Sending model to device {device}..."):
        model.to(device)
    with phase("Running model..."):
        with torch.no_grad():
            preds = model(**tokenized)
    with phase("Computing probabilities"):
        logits = preds.logits.detach()[mask_locations]
        probs = logits.softmax(dim=1)

    prob_sums = {}
    for pred_gender in GENDERS:
        with phase(f"Summing probabilities for {pred_gender}..."):
            probs_for_gender = probs[:, gender_masks[pred_gender]]
            prob_sums[pred_gender] = probs_for_gender.sum(dim=1)
    with phase("Saving probability sums..."):
        for row, femn_prob_sum, masc_prob_sum in zip(out, prob_sums['femn'], prob_sums['masc']):
            row['femn_prob_sum'] = femn_prob_sum.item()
            row['masc_prob_sum'] = masc_prob_sum.item()

    with phase("Writing output file..."):
        with open(out_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(out)

p = argparse.ArgumentParser()
p.add_argument("nouns", type=pathlib.Path)
p.add_argument("outfile", type=pathlib.Path)
p.add_argument("--verbs", type=pathlib.Path)
p.add_argument("--adjectives", type=pathlib.Path)
p.add_argument("--model", type=str, default="ai-forever/ruBert-large")
p.add_argument("--cache_dir", type=pathlib.Path, default='.cache')
p.add_argument("--max_len", type=int, default=2)

args = p.parse_args()

model = AutoModelForMaskedLM.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)
morph = pymorphy2.MorphAnalyzer(lang='ru')

with open(args.nouns, 'r') as f:
    nouns = [noun.strip() for noun in f]

if args.verbs:
    with open(args.verbs, 'r') as f:
        verbs = [verb.strip() for verb in f]
    check_masked_targets_multi(
        "{mask} {noun} вчера {fixed_target}.",
        nouns=nouns,
        fixed_targets=verbs,
        target_grammemes=PAST_VERB_FEATS,
        mask_grammemes=ADJ_FEATS,
        max_len=args.max_len,
        model=model,
        tokenizer=tokenizer,
        out_file=args.outfile
        )

if args.adjectives:
    with open(args.adjectives, 'r') as f:
        adjectives = [adjective.strip() for adjective in f]
    check_masked_targets_multi(
        "{fixed_target} {noun} вчера {mask}.",
        nouns=nouns,
        fixed_targets=adjectives,
        target_grammemes=ADJ_FEATS,
        mask_grammemes=PAST_VERB_FEATS,
        max_len=args.max_len,
        model=model,
        tokenizer=tokenizer,
        out_file=args.outfile
    )