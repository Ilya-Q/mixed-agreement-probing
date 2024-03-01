import masking
import pandas as pd
from transformers import AutoTokenizer

def test_simple_pattern():
    df = pd.DataFrame.from_dict({
        'pattern': ['{fixed_target} {noun} {mask} решительно.'],
        'masc': ['действовал'],
        'femn': ['действовала'],
    })
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruRoberta-large")
    applied = masking.make_subword_patterns(df, tokenizer)
    assert len(applied) == 1
    row = applied.iloc[0]
    assert row['pattern'] == '{fixed_target} {noun} действ{mask} решительно.'
    assert row['masc'] == 1613
    assert row['femn'] == 3599
