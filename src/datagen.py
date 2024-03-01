import pymorphy2
import transformers
import pandas as pd
import re
from typing import List, Set, Dict, Tuple
from collections import OrderedDict
from .common import ADJ_FEATS, PAST_VERB_FEATS, GENDERS

morph = pymorphy2.MorphAnalyzer(lang='ru')

def _sub_with_positions(pattern: str, **subs: Dict[str, str]):
    sub_positions = OrderedDict()
    pieces = re.split(r'\{(\w+)\}', pattern)

    cur = 0
    for i in range(len(pieces)):
        piece = pieces[i]
        if i%2 == 0:
            cur += len(piece)
        else:
            sub = subs[piece]
            pieces[i] = sub
            if piece in sub_positions:
                raise ValueError(f"Multiple occurrences of field '{piece}' in input string")
            sub_positions[piece] = (cur, cur + len(sub))
            cur += len(sub)
    return "".join(pieces), sub_positions 

def make_dataset_base(nouns: List[str], targets: List[str], target_grammemes: Set[str]) -> pd.DataFrame:
    parsed_targets = []
    for t in targets:
        try:
            parsed_targets.append(next(p for p in morph.parse(t) if target_grammemes in p.tag))
        except StopIteration:
            raise ValueError(f"Target {t} could not be parsed with grammeme set {target_grammemes}")
        
    return pd.DataFrame({"noun":nouns}).merge(
        pd.DataFrame({"fixed_target":parsed_targets}),
        how='cross'
    ).merge(
        pd.DataFrame({"fixed_target_gender":GENDERS}),
        how='cross'
    )

def apply_pattern(dataset: pd.DataFrame, pattern: str, mask: str) -> pd.DataFrame:
    def row_transform(row):
        sent, sub_positions = _sub_with_positions(pattern,
            noun=row['noun'],
            mask=mask,
            fixed_target=row["fixed_target"].inflect({row["fixed_target_gender"]}).word
        )
        sent = sent[0].upper() + sent[1:]
        return {"sent": sent, "sub_positions": sub_positions}
    ret = dataset.merge(dataset.apply(row_transform, axis='columns', result_type='expand'), left_index=True, right_index=True)
    ret["fixed_target"] = ret["fixed_target"].apply(lambda p: p.word)
    return ret

