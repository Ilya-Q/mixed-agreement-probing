import argparse
import pathlib
import csv
import transformers
import pymorphy2
import numpy as np
import torch
import itertools
import functools
from time import time
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Dict, Set

device='cpu'

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

def build_token_mask(tokenizer: transformers.PreTrainedTokenizer, grammemes: Set[str]):
    mask = np.zeros(len(tokenizer.get_vocab()), dtype=bool)
    for token_id in tokenizer.get_vocab().values():
        word = tokenizer.decode([token_id])
        word = word.strip().lstrip('_ ')
        if not word.startswith("##") and morph.word_is_known(word) and any(grammemes in parse.tag for parse in morph.parse(word)):
            mask[token_id] = True
    return mask

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
    genders = ('femn', 'masc')

    gender_masks: Dict[str, np.array] = {}
    for gender in genders:
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

    grammemes_for_gender = {gender: target_grammemes | {gender} for gender in genders}
    with phase(f"Building test sentences..."):
        for noun, fixed, fixed_gender in itertools.product(nouns, fixed_targets, genders):
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
    for pred_gender in genders:
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

args = p.parse_args()

model = AutoModelForMaskedLM.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)
morph = pymorphy2.MorphAnalyzer(lang='ru')

with open(args.nouns, 'r') as f:
    nouns = [noun.strip() for noun in f]

if args.verbs:
    with open(args.verbs, 'r') as f:
        verbs = [verb.strip() for verb in f]
    check_masked_targets(
        "{mask} {noun} вчера {fixed_target}.",
        nouns=nouns,
        fixed_targets=verbs,
        target_grammemes={'VERB', 'past', 'sing'},
        mask_grammemes={"ADJF", "nomn"},
        model=model,
        tokenizer=tokenizer,
        out_file=args.outfile
        )

if args.adjectives:
    with open(args.adjectives, 'r') as f:
        adjectives = [adjective.strip() for adjective in f]
    check_masked_targets(
        "{fixed_target} {noun} вчера {mask}.",
        nouns=nouns,
        fixed_targets=adjectives,
        target_grammemes={"ADJF", "nomn"},
        mask_grammemes={'VERB', 'past', 'sing'},
        model=model,
        tokenizer=tokenizer,
        out_file=args.outfile
    )