import click
from typing import Tuple
from pathlib import Path
from dataclasses import dataclass
import transformers
import pandas as pd
from .datagen import make_dataset_base, apply_pattern
from .masking import make_subword_patterns, make_simple_probs_map, make_whole_token_probs_map, mask
from .pred import apply_model
from .common import ADJ_FEATS, PAST_VERB_FEATS, GENDERS

FEATS = {
    "a": ADJ_FEATS,
    "v": PAST_VERB_FEATS,
}

@dataclass
class ExperimentBase:
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    base_data: pd.DataFrame
    output_dir: Path
    experiment_name: str
    target_type: str

@click.group()
@click.argument('nouns', type=click.Path(file_okay=True, dir_okay=False, exists=True, readable=True, path_type=Path))
@click.option('--results_folder', type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default='./results')
@click.option('--model', default="ai-forever/ruBert-large", type=click.STRING)
@click.option('--per_layer', default=False, is_flag=True)
@click.option('--targets', required=True, type=(
    click.Choice(['a', 'v'], case_sensitive=False),
    click.Path(file_okay=True, dir_okay=False, exists=True, readable=True, path_type=Path)
))
@click.pass_context
def cli(ctx: click.Context, nouns: Path, results_folder: Path, model: str, per_layer: bool, targets: Tuple[str, Path]):
    model = transformers.AutoModelForMaskedLM.from_pretrained(model, output_hidden_states=per_layer)
    if model.name_or_path.startswith("ai-forever/ruBert"):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model.name_or_path, do_lower_case=False, strip_accents=False)
        print("Using buggy tokenizer workaround")
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model.name_or_path)
    output_dir = Path(results_folder, model.name_or_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    target_type, target_lemmas = targets
    with nouns.open('r') as nouns_file, target_lemmas.open('r') as target_file:
        base_data = make_dataset_base(
            nouns=[noun.strip() for noun in nouns_file],
            targets=[target.strip() for target in target_file],
            target_grammemes=FEATS[target_type]
        )
    ctx.obj = ExperimentBase(
        model=model,
        tokenizer=tokenizer,
        base_data=base_data,
        output_dir=output_dir,
        experiment_name=f"{nouns.stem}-{target_lemmas.stem}{'-per_layer' if per_layer else ''}",
        target_type=target_type
    )

@cli.command()
@click.argument('patterns', type=click.Path(file_okay=True, dir_okay=False, exists=True, readable=True, path_type=Path))
@click.make_pass_decorator(ExperimentBase)
def subword(base: ExperimentBase, patterns: Path):
    patterns_parsed = make_subword_patterns(pd.read_csv(patterns), base.tokenizer)
    out = []
    for _, row in patterns_parsed.iterrows():
        pattern = row['pattern']
        probs_map = make_simple_probs_map(row.drop('pattern'))
        dataset = mask(base.base_data, pattern, base.tokenizer)
        results = apply_model(dataset, base.model, probs_map)
        results['pattern'] = pattern
        out.append(pd.concat((dataset, results), axis=1))
    output = base.output_dir / (base.experiment_name + f"-subword-{patterns.stem}.pkl")
    pd.concat(out, ignore_index=True).to_pickle(output)

# maps from the type of the fixed target to the pattern and the mask's features
PATTERNS = {
    "a": ("{fixed_target} {noun} вчера {mask}.", PAST_VERB_FEATS),
    "v": ("{mask} {noun} вчера {fixed_target}.", ADJ_FEATS),
}

@cli.command()
@click.option('--num_tokens', type=click.INT, default=1)
@click.make_pass_decorator(ExperimentBase)
def wholetoken(base: ExperimentBase, num_tokens: int):
    pattern, feats = PATTERNS[base.target_type]
    probs_map = make_whole_token_probs_map(base.tokenizer, {
        gender: frozenset({gender} | feats)
        for gender in GENDERS
    }, mask_length=num_tokens)
    dataset = mask(base.base_data, pattern, base.tokenizer, mask_length=num_tokens)
    results = apply_model(dataset, base.model, probs_map)
    results['pattern'] = pattern
    output = base.output_dir / (base.experiment_name + f"-wholetoken-{num_tokens}.pkl")
    pd.concat((dataset, results), axis=1).to_pickle(output)

if __name__ == '__main__':
    cli()