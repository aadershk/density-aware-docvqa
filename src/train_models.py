"""Train baseline (control) and density-aware LayoutLM. Usage: --task baseline | density | density_subset."""

import argparse
import gc
import json
import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from transformers import (
    LayoutLMTokenizerFast,
    LayoutLMForQuestionAnswering,
    LayoutLMConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    PreTrainedTokenizerBase,
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
TRAIN_SUBSET_CACHE = CACHE_DIR / "train_subset"
VAL_CACHE  = CACHE_DIR / "val"
TRAIN_JSON = PREPARED_DIR / "train_v1.0_prepared.json"
SUBSET_JSON = PREPARED_DIR / "train_v1.0_subset_25.json"
OUTPUT_BASELINE = BASE_DIR / "outputs" / "baseline_experiment"
OUTPUT_DENSITY  = BASE_DIR / "outputs" / "subset_experiment"

MODEL_NAME = "microsoft/layoutlm-base-uncased"
MAX_SEQ_LENGTH   = 512
DATALOADER_NUM_WORKERS = 0
DATALOADER_PIN_MEMORY  = False
SUBSET_RATIO = 0.25

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

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

class BaselineLayoutLM(LayoutLMForQuestionAnswering):
    """Baseline: same interface as density model but ignores density_scores."""

    def __init__(self, config: LayoutLMConfig):
        super().__init__(config)

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


class DensityAwareLayoutLM(LayoutLMForQuestionAnswering):
    """Density model: per-token density projection added before encoder."""

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


@dataclass
class DocVQADataCollator:
    """Tokenize raw JSON and build batches for saving to disk (cache step)."""
    tokenizer: PreTrainedTokenizerBase
    max_length: int = MAX_SEQ_LENGTH

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._process_batch(features)

    def _process_batch(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_input_ids, batch_attention_mask, batch_token_type_ids = [], [], []
        batch_bbox, batch_density_scores, batch_start_positions, batch_end_positions, batch_metadata = [], [], [], [], []
        for feature in features:
            processed = self._process_single(feature)
            batch_input_ids.append(processed["input_ids"])
            batch_attention_mask.append(processed["attention_mask"])
            batch_token_type_ids.append(processed["token_type_ids"])
            batch_bbox.append(processed["bbox"])
            batch_density_scores.append(processed["density_scores"])
            batch_start_positions.append(processed["start_positions"])
            batch_end_positions.append(processed["end_positions"])
            batch_metadata.append(processed["metadata"])
        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "token_type_ids": torch.stack(batch_token_type_ids),
            "bbox": torch.stack(batch_bbox),
            "density_scores": torch.stack(batch_density_scores),
            "start_positions": torch.tensor(batch_start_positions, dtype=torch.long),
            "end_positions": torch.tensor(batch_end_positions, dtype=torch.long),
            "metadata": batch_metadata,
        }

    def _process_single(self, feature: Dict[str, Any]) -> Dict[str, Any]:
        question = feature.get("question", "") or ""
        words = feature.get("words", []) or []
        boxes = feature.get("boxes", []) or []
        answers = feature.get("answers", []) or []
        density_scores = feature.get("token_density_scores", []) or []
        density_group = feature.get("density_group", "Unknown")
        if not words:
            words, boxes, density_scores = ["[EMPTY]"], [[0, 0, 0, 0]], [0.0]
        if len(density_scores) < len(words):
            density_scores += [0.0] * (len(words) - len(density_scores))
        if len(boxes) < len(words):
            boxes += [[0, 0, 0, 0]] * (len(words) - len(boxes))
        q_tokens = self.tokenizer(question, add_special_tokens=False)["input_ids"][:64]
        w_tokens, w_boxes, w_density = [], [], []
        for w, b, d in zip(words, boxes, density_scores):
            toks = self.tokenizer(w, add_special_tokens=False)["input_ids"]
            if toks:
                w_tokens.extend(toks)
                w_boxes.extend([b] * len(toks))
                w_density.extend([float(d)] * len(toks))
        specials = 3
        ctx_len = self.max_length - len(q_tokens) - specials
        w_tokens, w_boxes, w_density = w_tokens[:ctx_len], w_boxes[:ctx_len], w_density[:ctx_len]
        cls_id, sep_id, pad_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id
        input_ids = [cls_id] + q_tokens + [sep_id] + w_tokens + [sep_id]
        token_type_ids = [0] * (len(q_tokens) + 2) + [1] * (len(w_tokens) + 1)
        mask = [1] * len(input_ids)
        null_box = [0, 0, 0, 0]
        bbox = [null_box] * (len(q_tokens) + 2) + w_boxes + [null_box]
        density = [0.0] * (len(q_tokens) + 2) + w_density + [0.0]
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [pad_id] * pad_len
            mask += [0] * pad_len
            token_type_ids += [0] * pad_len
            bbox += [null_box] * pad_len
            density += [0.0] * pad_len
        start, end = 0, 0
        if answers:
            ctx_start = len(q_tokens) + 2
            for ans in answers:
                ans_toks = self.tokenizer(ans, add_special_tokens=False)["input_ids"]
                if not ans_toks:
                    continue
                for i in range(ctx_start, len(input_ids) - len(ans_toks) + 1):
                    if input_ids[i : i + len(ans_toks)] == ans_toks:
                        start, end = i, i + len(ans_toks) - 1
                        break
                if start:
                    break
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "bbox": torch.tensor(bbox, dtype=torch.long),
            "density_scores": torch.tensor(density, dtype=torch.float),
            "start_positions": start,
            "end_positions": end,
            "metadata": {"density_group": str(density_group), "answers": answers, "question": str(question)},
        }


@dataclass
class CachedDataCollator:
    """Stack cached tensors for training (includes start/end positions)."""
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            return {}
        TENSOR_KEYS = ["input_ids", "attention_mask", "token_type_ids", "bbox", "density_scores", "start_positions", "end_positions"]
        batch = {}
        first = features[0]
        for key in TENSOR_KEYS:
            if key in first:
                items = [f[key] for f in features]
                batch[key] = torch.stack(items) if isinstance(items[0], torch.Tensor) else torch.tensor(items)
        batch["metadata"] = [
            {"density_group": f.get("density_group", "Unknown"), "answers": f.get("answers", [])}
            for f in features
        ]
        return batch


class MemoryCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            logger.info(f"Step {state.global_step} | VRAM: {get_vram_usage():.1f}MB")
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        clear_memory()


class BaselineTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs.pop("metadata", None)
        allowed = ["input_ids", "attention_mask", "token_type_ids", "bbox", "density_scores", "start_positions", "end_positions"]
        clean = {k: v for k, v in inputs.items() if k in allowed}
        outputs = model(**clean)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


class DensityAwareTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs.pop("metadata", None)
        allowed = ["input_ids", "attention_mask", "token_type_ids", "bbox", "density_scores", "start_positions", "end_positions"]
        clean = {k: v for k, v in inputs.items() if k in allowed}
        outputs = model(**clean)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


def evaluate_stratified(model, loader, tokenizer, device):
    """Stratified ANLS on val set (Sparse/Medium/Dense/Overall)."""
    model.eval()
    results = {"Sparse": [], "Medium": [], "Dense": [], "Overall": []}
    allowed = ["input_ids", "attention_mask", "token_type_ids", "bbox", "density_scores"]
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            meta = batch.pop("metadata", [])
            # print(batch['bbox'].shape)  # debug
            clean_batch = {k: v.to(device) for k, v in batch.items() if k in allowed}
            outputs = model(**clean_batch)
            start_logits = outputs.start_logits.cpu().numpy()
            end_logits = outputs.end_logits.cpu().numpy()
            ids = clean_batch["input_ids"].cpu().numpy()
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

def run_baseline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = load_from_disk(str(TRAIN_SUBSET_CACHE))
    val_dataset = load_from_disk(str(VAL_CACHE))
    tokenizer = LayoutLMTokenizerFast.from_pretrained(MODEL_NAME)
    config = LayoutLMConfig.from_pretrained(MODEL_NAME)
    model = BaselineLayoutLM.from_pretrained(MODEL_NAME, config=config)
    model.to(device)
    model.gradient_checkpointing_enable()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=1,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=DATALOADER_PIN_MEMORY,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch",
    )
    trainer = BaselineTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CachedDataCollator(tokenizer),
        callbacks=[MemoryCleanupCallback()],
    )
    try:
        trainer.train()
        model.save_pretrained(Path(args.output_dir) / "final_model")
        tokenizer.save_pretrained(Path(args.output_dir) / "final_model")
    except Exception as e:
        logger.warning(f"Training interrupted: {e}")
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_train_batch_size * 2,
        collate_fn=CachedDataCollator(tokenizer),
    )
    metrics = evaluate_stratified(model, val_loader, tokenizer, device)
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}%")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


# TODO: try with a learning rate scheduler next time
def run_density(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = load_from_disk(str(TRAIN_SUBSET_CACHE))
    val_dataset = load_from_disk(str(VAL_CACHE))
    tokenizer = LayoutLMTokenizerFast.from_pretrained(MODEL_NAME)
    config = LayoutLMConfig.from_pretrained(MODEL_NAME)
    model = DensityAwareLayoutLM.from_pretrained(MODEL_NAME, config=config)
    model.to(device)
    model.gradient_checkpointing_enable()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=1,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=DATALOADER_PIN_MEMORY,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch",
    )
    trainer = DensityAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CachedDataCollator(tokenizer),
        callbacks=[MemoryCleanupCallback()],
    )
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        model.save_pretrained(Path(args.output_dir) / "final_model")
        tokenizer.save_pretrained(Path(args.output_dir) / "final_model")
    except Exception as e:
        logger.warning(f"Training interrupted: {e}")
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_train_batch_size * 2,
        collate_fn=CachedDataCollator(tokenizer),
    )
    metrics = evaluate_stratified(model, val_loader, tokenizer, device)
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}%")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

def create_stratified_subset():
    if SUBSET_JSON.exists():
        logger.info("Subset JSON already exists.")
        return
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            data = data["data"]
    groups = {"Sparse": [], "Medium": [], "Dense": [], "Unknown": []}
    for item in data:
        g = item.get("density_group", "Unknown")
        groups[g].append(item)
    subset_data = []
    for g, items in groups.items():
        count = int(len(items) * SUBSET_RATIO)
        selected = random.sample(items, count)
        subset_data.extend(selected)
        print(f"  {g}: Selected {count}/{len(items)}")
    with open(SUBSET_JSON, "w", encoding="utf-8") as f:
        json.dump({"data": subset_data}, f)


def cache_subset():
    if TRAIN_SUBSET_CACHE.exists():
        logger.info("Subset cache exists. Skipping tokenization.")
        return
    tokenizer = LayoutLMTokenizerFast.from_pretrained(MODEL_NAME)
    collator = DocVQADataCollator(tokenizer, max_length=MAX_SEQ_LENGTH)

    def process_batch(batch):
        keys = list(batch.keys())
        examples = [dict(zip(keys, v)) for v in zip(*batch.values())]
        processed = collator._process_batch(examples)
        output = {k: v.tolist() if torch.is_tensor(v) else v for k, v in processed.items()}
        meta = processed["metadata"]
        output["density_group"] = [m["density_group"] for m in meta]
        output["answers"] = [m["answers"] for m in meta]
        output.pop("metadata", None)
        return output

    ds = load_dataset("json", data_files={"train": str(SUBSET_JSON)}, field="data")
    ds["train"].map(process_batch, batched=True, batch_size=32, remove_columns=ds["train"].column_names).save_to_disk(
        str(TRAIN_SUBSET_CACHE)
    )


def run_density_subset(args):
    create_stratified_subset()
    cache_subset()
    run_density(args)


def main():
    parser = argparse.ArgumentParser(description="Train baseline or density model")
    parser.add_argument("--task", type=str, required=True, choices=["baseline", "density", "density_subset"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="For density/density_subset only")
    args = parser.parse_args()

    if args.task == "baseline":
        args.output_dir = args.output_dir or str(OUTPUT_BASELINE)
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print("Loading cached data (train_subset, val)...")
        run_baseline(args)
    elif args.task == "density":
        args.output_dir = args.output_dir or str(OUTPUT_DENSITY)
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print("Loading cached data (train_subset, val)...")
        run_density(args)
    elif args.task == "density_subset":
        args.output_dir = args.output_dir or str(OUTPUT_DENSITY)
        print("Creating stratified subset, caching, then training density (resume if --resume_from_checkpoint).")
        run_density_subset(args)


if __name__ == "__main__":
    main()
