import conllu
import argparse
import pathlib
import collections
import itertools
import contextlib
import json
import sys
import multiprocessing

p = argparse.ArgumentParser(description="Retrieve adjectives and verbs to go with nouns")
p.add_argument("--nouns", type=pathlib.Path, required=False)
p.add_argument("--count", type=int, default=5)
p.add_argument("sources", type=pathlib.Path, nargs="+")
p.add_argument("--intransitive", action="store_true")

args = p.parse_args()
if args.nouns is not None:
    nouns, verbs, adjectives = set(), {}, {}
    with open(args.noun_list, "r") as f:
        for noun in f:
            noun = noun.strip()
            nouns.add(noun)
            verbs[noun] = collections.Counter()
            adjectives[noun] = collections.Counter()
else:
    verbs, adjectives = collections.Counter(), collections.Counter()

with contextlib.ExitStack() as stack:
    sources = itertools.chain.from_iterable(
        map(
            lambda f: conllu.parse_incr(
                stack.enter_context(
                    open(f, "r")
                    )
                ),
                args.sources
            )
        )
    if args.nouns is not None:
        for sentence in sources:
            targets = [token for token in sentence if token["lemma"] in nouns]
            for target in targets:
                adjectives[target["lemma"]].update(token["lemma"] for token in sentence if token["head"] == target["id"] and token["deprel"] == 'amod' and token["upos"] == "ADJ")
                if target["deprel"] == 'nsubj':
                    candidates = filter(lambda token: token["upos"] == "VERB", sentence)
                    if args.intransitive:
                        transitives = {token["head"] for token in sentence if token["deprel"] in {"obj", "iobj", "xcomp"}}
                        candidates = filter(lambda token: token["id"] not in transitives, candidates)
                    verbs[target["lemma"]].update(token["lemma"] for token in candidates if token["head"] == target["id"] and token["feats"]["Voice"] == 'Act')
    else:
        for sentence in sources:
            candidates = filter(lambda token: token["upos"] == "VERB", sentence)
            if args.intransitive:
                transitives = {token["head"] for token in sentence if token["deprel"] in {"obj", "iobj", "xcomp"}}
                candidates = filter(lambda token: token["id"] not in transitives, candidates)
            verbs.update(token["lemma"] for token in candidates)
            adjectives.update(token["lemma"] for token in sentence if token["upos"] == "ADJ")

if args.nouns is not None:
    json.dump([{"noun": noun, "adjectives": adjectives[noun].most_common(args.count), "verbs": verbs[noun].most_common(args.count)} for noun in nouns], sys.stdout, ensure_ascii=False)
else:
    json.dump({"adjectives": adjectives.most_common(args.count), "verbs": verbs.most_common(args.count)}, sys.stdout, ensure_ascii=False)