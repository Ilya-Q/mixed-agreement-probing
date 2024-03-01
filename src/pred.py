import transformers
import pandas as pd
import torch
from typing import Callable

@torch.no_grad()
def apply_model(
        dataset: pd.DataFrame,
        model: transformers.PreTrainedModel,
        probs_map: Callable[[torch.Tensor], pd.DataFrame],
        batch_size=100) -> pd.DataFrame:
    model.eval()
    per_layer = model.config.output_hidden_states
    lm_head = model.cls if hasattr(model, 'cls') else model.lm_head
    ret = []
    for batch_start in range(0, len(dataset), batch_size):
        batch = dataset.iloc[batch_start:batch_start+batch_size]
        preds = model(
            input_ids=torch.tensor(batch['input_ids'].to_list()),
            attention_mask=torch.tensor(batch['attention_mask'].to_list())
        )
        batch_mask = torch.tensor(batch["mask_locations"].to_list())
        if per_layer:
            per_layer_probs = (lm_head(hidden_state)[batch_mask].softmax(dim=1) for hidden_state in preds.hidden_states)
            ret.append(pd.concat(map(probs_map, per_layer_probs)).groupby(level=0).agg(list))
        else:
            probs = preds.logits[batch_mask].softmax(dim=1)
            ret.append(probs_map(probs))
    return pd.concat(ret, ignore_index=True)

