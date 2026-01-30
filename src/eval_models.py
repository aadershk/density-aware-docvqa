"""Run all four evals (Untrained, Baseline, Density, SOTA). Writes outputs/FINAL_RESULTS.json."""

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LayoutLMTokenizerFast,
    LayoutLMForQuestionAnswering,
    LayoutLMConfig,
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent.resolve()
DATA_DIR   = BASE_DIR / "Data"
PREPARED_DIR = DATA_DIR / "prepared"
CACHE_DIR  = DATA_DIR / "cached"
VAL_CACHE  = CACHE_DIR / "val"
VAL_JSON   = PREPARED_DIR / "val_v1.0_prepared.json"
OUTPUT_BASELINE = BASE_DIR / "outputs" / "baseline_experiment"
OUTPUT_DENSITY  = BASE_DIR / "outputs" / "subset_experiment"
FINAL_RESULTS_JSON = BASE_DIR / "outputs" / "FINAL_RESULTS.json"

MODEL_NAME = "microsoft/layoutlm-base-uncased"
MAX_SEQ_LENGTH     = 512
BATCH_SIZE_CACHED  = 16
BATCH_SIZE_IMPIRA  = 8

EXPERIMENTS = [
    {"name": "Untrained", "data_type": "cached", "model_path": MODEL_NAME, "model_type": "standard"},
    {"name": "Baseline", "data_type": "cached", "model_path": str(OUTPUT_BASELINE / "final_model"), "model_type": "standard"},
    {"name": "Density", "data_type": "cached", "model_path": str(OUTPUT_DENSITY / "final_model"), "model_type": "density"},
    {"name": "SOTA", "data_type": "impira", "model_path": "impira/layoutlm-document-qa", "model_type": "impira"},
]

def anls_score(pred: str, gts: List[str], threshold: float = 0.5) -> float:
    """ANLS between pred and gts (threshold 0.5)."""
    if not gts:
        return 0.0
    pred = pred.lower().strip()

    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    max_score = 0.0
    for gt in gts:
        gt = gt.lower().strip()
        if not gt and not pred:
            return 1.0
        if not gt:
            continue
        dist = levenshtein(pred, gt)
        length = max(len(pred), len(gt))
        nl_dist = dist / length
        score = 1.0 - nl_dist if nl_dist < threshold else 0.0
        max_score = max(max_score, score)
    return max_score

class DensityAwareLayoutLM(LayoutLMForQuestionAnswering):
    """Same as train_models so we can load the density checkpoint."""

    def __init__(self, config: LayoutLMConfig):
        super().__init__(config)
        self.density_projection = nn.Linear(1, config.hidden_size)
        nn.init.normal_(self.density_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.density_projection.bias)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        density_scores=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None:
            device = input_ids.device
        else:
            device = inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.shape, dtype=torch.long, device=device)
        if inputs_embeds is None:
            inputs_embeds = self.layoutlm.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )
        if density_scores is not None:
            density_scores = density_scores.to(device, dtype=inputs_embeds.dtype)
            density_embeds = self.density_projection(density_scores.unsqueeze(-1))
            inputs_embeds = inputs_embeds + density_embeds
        outputs = self.layoutlm.encoder(
            inputs_embeds,
            attention_mask=attention_mask.unsqueeze(1).unsqueeze(2),
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CachedCollatorEval:
    """Stack cached tensors for eval (no start/end positions)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        if not features:
            return {}
        batch = {}
        for key in ["input_ids", "attention_mask", "token_type_ids", "bbox", "density_scores"]:
            if key in features[0]:
                items = [f[key] for f in features]
                batch[key] = torch.stack(items) if isinstance(items[0], torch.Tensor) else torch.tensor(items)
        batch["metadata"] = [
            {"density_group": f.get("density_group", "Unknown"), "answers": f.get("answers", [])}
            for f in features
        ]
        return batch


class ImpiraDataset(Dataset):
    """Raw JSON + manual tokenization for Impira (tokenizer mismatch with cache)."""

    def __init__(self, json_path, tokenizer, max_length=MAX_SEQ_LENGTH, type_vocab_size=2):
        with open(json_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            self.data = content["data"] if isinstance(content, dict) and "data" in content else content
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = len(tokenizer)
        self.type_vocab_size = type_vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = str(item.get("question", "") or "").strip() or "[UNK]"
        raw_words = item.get("words", []) or []
        if not isinstance(raw_words, (list, tuple)):
            raw_words = [raw_words]
        words = [str(w) if w is not None else "" for w in raw_words]
        if len(words) == 0:
            words, boxes = ["[EMPTY]"], [[0, 0, 0, 0]]
        else:
            boxes_raw = item.get("boxes", []) or []
            if not isinstance(boxes_raw, (list, tuple)):
                boxes_raw = [boxes_raw]
            boxes = []
            for b in boxes_raw:
                if b is None:
                    box = [0, 0, 0, 0]
                elif isinstance(b, (list, tuple)) and len(b) >= 4:
                    box = [max(0, min(1000, int(float(x)))) for x in b[:4]]
                else:
                    box = [0, 0, 0, 0]
                boxes.append(box)
            min_len = min(len(words), len(boxes))
            words, boxes = words[:min_len], boxes[:min_len]
            if len(boxes) < len(words):
                boxes.extend([[0, 0, 0, 0]] * (len(words) - len(boxes)))
        try:
            q_enc = self.tokenizer(question, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
            question_tokens = [int(t) for t in q_enc["input_ids"] if 0 <= int(t) < self.vocab_size][:64]
        except Exception:
            return None
        word_tokens, word_boxes_expanded = [], []
        for word, box in zip(words, boxes):
            if not word or not isinstance(word, str):
                continue
            try:
                w_enc = self.tokenizer(word, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
                subtokens = [int(t) for t in w_enc["input_ids"] if 0 <= int(t) < self.vocab_size]
            except Exception:
                continue
            if not subtokens:
                continue
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                box = [0, 0, 0, 0]
            else:
                box = [max(0, min(1000, int(float(x)))) for x in box[:4]]
            word_tokens.extend(subtokens)
            word_boxes_expanded.extend([box] * len(subtokens))
        if not word_tokens:
            unk_id = getattr(self.tokenizer, "unk_token_id", None) or 100
            word_tokens = [unk_id if 0 <= unk_id < self.vocab_size else 0]
            word_boxes_expanded = [[0, 0, 0, 0]]
        special_count = 3
        max_ctx = max(1, self.max_length - len(question_tokens) - special_count)
        if len(word_tokens) > max_ctx:
            word_tokens = word_tokens[:max_ctx]
            word_boxes_expanded = word_boxes_expanded[:max_ctx]
        cls_id = getattr(self.tokenizer, "cls_token_id", None) or getattr(self.tokenizer, "bos_token_id", None)
        if cls_id is None or not (0 <= cls_id < self.vocab_size):
            cls_id = 0 if 0 < self.vocab_size else 101
        sep_id = getattr(self.tokenizer, "sep_token_id", None) or getattr(self.tokenizer, "eos_token_id", None)
        if sep_id is None or not (0 <= sep_id < self.vocab_size):
            sep_id = 2 if 2 < self.vocab_size else 102
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None or not (0 <= pad_id < self.vocab_size):
            pad_id = 1 if 1 < self.vocab_size else 0
        input_ids = [cls_id] + question_tokens + [sep_id] + word_tokens + [sep_id]
        input_ids = [max(0, min(self.vocab_size - 1, int(t))) for t in input_ids]
        if self.type_vocab_size == 1:
            token_type_ids = [0] * len(input_ids)
        else:
            token_type_ids = [0] * (len(question_tokens) + 2) + [1] * (len(word_tokens) + 1)
        max_tt = max(0, self.type_vocab_size - 1)
        token_type_ids = [max(0, min(max_tt, int(t))) for t in token_type_ids]
        null_box = [0, 0, 0, 0]
        bbox = [null_box] * (len(question_tokens) + 2) + word_boxes_expanded + [null_box]
        attention_mask = [1] * len(input_ids)
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [pad_id] * pad_len
            token_type_ids += [0] * pad_len
            attention_mask += [0] * pad_len
            bbox += [null_box] * pad_len
        input_ids = input_ids[: self.max_length]
        token_type_ids = token_type_ids[: self.max_length]
        attention_mask = attention_mask[: self.max_length]
        bbox = bbox[: self.max_length]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "bbox": torch.tensor(bbox, dtype=torch.long),
            "answers": item.get("answers", []),
            "density_group": item.get("density_group", "Unknown"),
        }


class ImpiraCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = [b for b in features if b is not None]
        if not batch:
            return {}
        out = {}
        for key in ["input_ids", "attention_mask", "token_type_ids", "bbox"]:
            if key in batch[0]:
                out[key] = torch.stack([b[key] for b in batch])
        out["metadata"] = [{"density_group": b.get("density_group", "Unknown"), "answers": b.get("answers", [])} for b in batch]
        return out


def get_eval_dataloader(experiment_type, tokenizer=None, type_vocab_size=2, cached_dataset=None):
    if experiment_type == "cached":
        if cached_dataset is None:
            cached_dataset = load_from_disk(str(VAL_CACHE))
        return DataLoader(cached_dataset, batch_size=BATCH_SIZE_CACHED, collate_fn=CachedCollatorEval(tokenizer))
    elif experiment_type == "impira":
        dataset = ImpiraDataset(VAL_JSON, tokenizer, max_length=MAX_SEQ_LENGTH, type_vocab_size=type_vocab_size)
        return DataLoader(dataset, batch_size=BATCH_SIZE_IMPIRA, collate_fn=ImpiraCollator(tokenizer))
    raise ValueError(f"Unknown experiment_type: {experiment_type}")


def load_model_and_tokenizer_for_eval(exp, device):
    path = exp["model_path"]
    model_type = exp["model_type"]
    if model_type == "impira":
        config = AutoConfig.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path, add_prefix_space=True)
        if getattr(config, "model_type", None) == "layoutlm":
            model = LayoutLMForQuestionAnswering.from_pretrained(path, config=config)
        else:
            from transformers import AutoModelForQuestionAnswering
            model = AutoModelForQuestionAnswering.from_pretrained(path, config=config)
        return model, tokenizer, getattr(config, "type_vocab_size", 2)
    elif model_type == "density":
        config = LayoutLMConfig.from_pretrained(path)
        model = DensityAwareLayoutLM.from_pretrained(path, config=config)
        tokenizer = LayoutLMTokenizerFast.from_pretrained(path)
        return model, tokenizer, None
    else:
        tokenizer = LayoutLMTokenizerFast.from_pretrained(path)
        model = LayoutLMForQuestionAnswering.from_pretrained(path)
        return model, tokenizer, None


def run_eval_loop(name, model, tokenizer, loader, device, model_type):
    results = {"Sparse": [], "Medium": [], "Dense": [], "Overall": []}
    with torch.no_grad():
        for batch in tqdm(loader, desc=name):
            if not batch:
                continue
            meta = batch.pop("metadata")
            batch = {k: v.to(device) for k, v in batch.items()}
            if model_type == "standard" and "density_scores" in batch:
                batch.pop("density_scores")
            if "bbox" in batch:
                batch["bbox"] = torch.clamp(batch["bbox"], 0, 1000)
            if "input_ids" in batch:
                batch["input_ids"] = torch.clamp(batch["input_ids"], 0, len(tokenizer) - 1)
            if "token_type_ids" in batch:
                tv = getattr(model.config, "type_vocab_size", 2)
                batch["token_type_ids"] = torch.clamp(batch["token_type_ids"], 0, max(0, tv - 1))
            outputs = model(**batch)
            start_logits = outputs.start_logits.cpu().numpy()
            end_logits = outputs.end_logits.cpu().numpy()
            ids = batch["input_ids"].cpu().numpy()
            # print(ids.shape)  # debug
            for i in range(len(ids)):
                s, e = np.argmax(start_logits[i]), np.argmax(end_logits[i])
                if e < s:
                    e = s
                pred = tokenizer.decode(ids[i][s : e + 1], skip_special_tokens=True).strip()
                score = anls_score(pred, meta[i]["answers"])
                grp = meta[i]["density_group"]
                if grp in results:
                    results[grp].append(score)
                results["Overall"].append(score)
    return {k: (np.mean(v) * 100 if v else 0.0) for k, v in results.items()}


def run_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    try:
        cached_dataset = load_from_disk(str(VAL_CACHE))
    except Exception as e:
        sys.exit(f"Error loading val cache: {e}")
    # FIXME: could parallelise model loads if we had more VRAM
    final_results = {}
    for exp in EXPERIMENTS:
        name = exp["name"]
        print(f"\n--- {name} ---")
        try:
            model, tokenizer, type_vocab_size = load_model_and_tokenizer_for_eval(exp, device)
            model.to(device)
            model.eval()
            if exp["data_type"] == "cached":
                loader = get_eval_dataloader("cached", tokenizer=tokenizer, cached_dataset=cached_dataset)
            else:
                loader = get_eval_dataloader("impira", tokenizer=tokenizer, type_vocab_size=type_vocab_size or 2)
            metrics = run_eval_loop(name, model, tokenizer, loader, device, exp["model_type"])
            final_results[name] = metrics
        except Exception as e:
            logger.exception(f"FAILED {name}: {e}")
            final_results[name] = {"Sparse": 0.0, "Medium": 0.0, "Dense": 0.0, "Overall": 0.0}
    print("\n" + "=" * 75)
    print("FINAL RESULTS — ANLS (%) by Density Group")
    print("=" * 75)
    print(f"{'Model':<14} | {'Sparse':>8} | {'Medium':>8} | {'Dense':>8} | {'Overall':>8}")
    print("-" * 75)
    for name, m in final_results.items():
        print(f"{name:<14} | {m['Sparse']:>8.2f} | {m['Medium']:>8.2f} | {m['Dense']:>8.2f} | {m['Overall']:>8.2f}")
    print("=" * 75)
    FINAL_RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_RESULTS_JSON, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved to: {FINAL_RESULTS_JSON}")


def main():
    run_eval()


if __name__ == "__main__":
    main()
