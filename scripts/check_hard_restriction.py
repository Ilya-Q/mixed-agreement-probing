import argparse
import pathlib
import csv
import transformers
import pymorphy2
import numpy as np
import itertools
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Dict

def check_masked_adjectives(nouns: List[str], verbs: List[str], model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer) -> None:
    verbs = [next(p for p in morph.parse(v) if {'INFN'} in p.tag) for v in verbs]
    #assert(all({'INFN'} in parse.tag for parse in verbs))

    adj_masks: Dict[str, np.array] = {}
    for gender in ('femn', 'masc'):
        adj_masks[gender] = np.zeros(len(tokenizer.get_vocab()), dtype=int)
        adj_masks[gender][[idx for word, idx in tokenizer.get_vocab().items() if not word.startswith("##") and any({"ADJF", gender, "nomn"} in parse.tag for parse in morph.parse(word))]] = 1

    out: List[Dict] = []
    #sents: List[str] = []
    fields = ["noun", "verb", "verb_gender", "femn_prob_sum", "masc_prob_sum"]

    for noun, verb, verb_gender in itertools.product(nouns, verbs, ('femn', 'masc')):
        row = { "noun": noun, "verb": verb.word, "verb_gender": verb_gender }

        infl_verb = verb.inflect({'VERB', 'past', 'sing', verb_gender}).word
        sent = f"{tokenizer.mask_token} {noun} вчера {infl_verb}."
        tokenized = tokenizer(sent, return_tensors='pt')
        assert(tokenized["input_ids"][0,1] == tokenizer.mask_token_id)
        preds = model(**tokenized)
        for pred_gender in ('femn', 'masc'):
            prob_sum = np.sum(preds.logits.detach()[:,1,:].flatten().softmax(dim=0).numpy() * adj_masks[pred_gender])
            row[f"{pred_gender}_prob_sum"] = prob_sum
        out.append(row)
        #print(row)

    
    with open("mask_adj.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    
def check_masked_verbs(nouns: List[str], adjectives: List[str], model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer) -> None:
    adjectives = [next(p for p in morph.parse(a) if {'ADJF', 'sing', 'masc', 'nomn'} in p.tag) for a in adjectives]
    assert(all({'ADJF', 'sing', 'masc', 'nomn'} in parse.tag for parse in adjectives))

    verb_masks: Dict[str, np.array] = {}
    for gender in ('femn', 'masc'):
        verb_masks[gender] = np.zeros(len(tokenizer.get_vocab()), dtype=int)
        verb_masks[gender][[idx for word, idx in tokenizer.get_vocab().items() if not word.startswith("##") and any({"VERB", gender, "past"} in parse.tag for parse in morph.parse(word))]] = 1

    out: List[Dict] = []
    #sents: List[str] = []
    fields = ["noun", "adjective", "adjective_gender", "femn_prob_sum", "masc_prob_sum"]

    for noun, adjective, adjective_gender in itertools.product(nouns, adjectives, ('femn', 'masc')):
        row = { "noun": noun, "adjective": adjective.word, "adjective_gender": adjective_gender }

        infl_adjective = adjective.inflect({'ADJF', 'sing', adjective_gender, 'nomn'}).word
        sent = f"{infl_adjective} {noun} вчера {tokenizer.mask_token}."
        tokenized = tokenizer(sent, return_tensors='pt')
        assert(tokenized["input_ids"][0,-3] == tokenizer.mask_token_id)
        preds = model(**tokenized)
        for pred_gender in ('femn', 'masc'):
            prob_sum = np.sum(preds.logits.detach()[:,-3,:].flatten().softmax(dim=0).numpy() * verb_masks[pred_gender])
            row[f"{pred_gender}_prob_sum"] = prob_sum
        out.append(row)
        #print(row)

    
    with open("mask_verb.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)

p = argparse.ArgumentParser()
p.add_argument("--model", type=str, default="ai-forever/ruBert-large")
p.add_argument("nouns", type=pathlib.Path)
p.add_argument("--verbs", type=pathlib.Path)
p.add_argument("--adjectives", type=pathlib.Path)

args = p.parse_args()

model = AutoModelForMaskedLM.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)
morph = pymorphy2.MorphAnalyzer(lang='ru')

with open(args.nouns, 'r') as f:
    nouns = [noun.strip() for noun in f]

if args.verbs:
    with open(args.verbs, 'r') as f:
        verbs = [verb.strip() for verb in f]
    check_masked_adjectives(nouns, verbs, model, tokenizer)

if args.adjectives:
    with open(args.adjectives, 'r') as f:
        adjectives = [adjective.strip() for adjective in f]
    check_masked_verbs(nouns, adjectives, model, tokenizer)