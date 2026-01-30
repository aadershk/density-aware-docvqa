"""Pre-process DocVQA into cached tensors so training doesn't bottleneck on CPU tokenization."""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import LayoutLMTokenizerFast

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent.resolve()
DATA_DIR   = BASE_DIR / "Data" / "prepared"
CACHE_DIR  = BASE_DIR / "Data" / "cached"

MODEL_NAME     = "microsoft/layoutlm-base-uncased"
MAX_SEQ_LENGTH = 512

TRAIN_SUBSET_JSONL = DATA_DIR / "train_v1.0_subset.jsonl"
TRAIN_SUBSET_JSON = DATA_DIR / "train_v1.0_subset.json"
VAL_FILE_JSONL    = DATA_DIR / "val_v1.0_prepared.jsonl"
VAL_FILE_JSON     = DATA_DIR / "val_v1.0_prepared.json"

TRAIN_CACHE_DIR = CACHE_DIR / "train"
VAL_CACHE_DIR   = CACHE_DIR / "val"


def find_answer_span(
    ids: List[int],
    answers: List[str],
    context_start: int,
    tokenizer: LayoutLMTokenizerFast
) -> Tuple[int, int]:
    """Returns (start, end) token indices for first matching answer in ids, or (0,0)."""
    if not answers or not isinstance(answers, list):
        return 0, 0

    for ans in answers:
        if not ans or not isinstance(ans, str):
            continue
        try:
            enc = tokenizer(ans, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
            ans_tokens = enc['input_ids']
            if len(ans_tokens) == 0:
                continue
            for i in range(context_start, len(ids) - len(ans_tokens) + 1):
                if ids[i:i + len(ans_tokens)] == ans_tokens:
                    return i, i + len(ans_tokens) - 1
        except Exception as e:
            logger.warning(f"Error tokenizing answer '{ans}': {e}")
            continue
    return 0, 0


def prepare_features(
    examples: Dict[str, List[Any]],
    tokenizer: LayoutLMTokenizerFast,
    max_length: int = MAX_SEQ_LENGTH
) -> Dict[str, List[Any]]:
    """Batch tokenize + align density/labels; mirrors DocVQADataCollator logic for cache consistency."""
    if 'question' not in examples:
        raise ValueError("Missing 'question' key in examples")
    n = len(examples['question'])
    if n == 0:
        raise ValueError("Empty batch")

    batch_ids = []
    batch_mask = []
    batch_ttids = []
    batch_bbox = []
    batch_density = []
    batch_start = []
    batch_end = []
    batch_density_groups = []
    batch_answers = []

    for i in range(n):
        question = examples.get('question', [''])[i] if i < len(examples.get('question', [])) else ''
        words = examples.get('words', [[]])[i] if i < len(examples.get('words', [])) else []
        boxes = examples.get('boxes', [[]])[i] if i < len(examples.get('boxes', [])) else []
        answers = examples.get('answers', [[]])[i] if i < len(examples.get('answers', [])) else []
        density_scores = examples.get('token_density_scores', [[]])[i] if i < len(examples.get('token_density_scores', [])) else []
        density_group = examples.get('density_group', ['Unknown'])[i] if i < len(examples.get('density_group', [])) else 'Unknown'

        if not isinstance(question, str):
            question = str(question) if question else ''
        if not isinstance(words, list):
            words = []
        if not isinstance(boxes, list):
            boxes = []
        if not isinstance(answers, list):
            answers = [answers] if answers else []
        if not isinstance(density_scores, list):
            density_scores = []
        if not isinstance(density_group, str):
            density_group = str(density_group) if density_group else 'Unknown'

        if not words:
            words = ['[EMPTY]']
            boxes = [[0, 0, 0, 0]]
            density_scores = [0.0]

        if len(density_scores) < len(words):
            density_scores = list(density_scores) + [0.0] * (len(words) - len(density_scores))
        elif len(density_scores) > len(words):
            density_scores = list(density_scores)[:len(words)]
        if len(boxes) < len(words):
            boxes = list(boxes) + [[0, 0, 0, 0]] * (len(words) - len(boxes))
        elif len(boxes) > len(words):
            boxes = list(boxes)[:len(words)]

        try:
            q_enc = tokenizer(question, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
            question_tokens = q_enc['input_ids']
        except Exception as e:
            logger.warning(f"Error tokenizing question: {e}, using empty question")
            question_tokens = []
        max_q_len = 64
        if len(question_tokens) > max_q_len:
            question_tokens = question_tokens[:max_q_len]

        word_tokens = []
        word_boxes_expanded = []
        word_density_expanded = []
        for word, box, density in zip(words, boxes, density_scores):
            if not word or not isinstance(word, str):
                continue
            try:
                w_enc = tokenizer(word, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
                subtokens = w_enc['input_ids']
            except Exception as e:
                logger.warning(f"Error tokenizing word '{word}': {e}, skipping")
                continue
            if len(subtokens) == 0:
                continue
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                box = [0, 0, 0, 0]
            else:
                box = [int(x) for x in box[:4]]
            try:
                density = float(density)
            except (ValueError, TypeError):
                density = 0.0
            word_tokens.extend(subtokens)
            word_boxes_expanded.extend([list(box)] * len(subtokens))
            word_density_expanded.extend([density] * len(subtokens))

        if not word_tokens:
            word_tokens = [tokenizer.unk_token_id]
            word_boxes_expanded = [[0, 0, 0, 0]]
            word_density_expanded = [0.0]

        special_tokens_count = 3
        max_context_length = max(1, max_length - len(question_tokens) - special_tokens_count)
        if len(word_tokens) > max_context_length:
            word_tokens = word_tokens[:max_context_length]
            word_boxes_expanded = word_boxes_expanded[:max_context_length]
            word_density_expanded = word_density_expanded[:max_context_length]

        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        null_box = [0, 0, 0, 0]

        input_ids = [cls_id] + question_tokens + [sep_id] + word_tokens + [sep_id]
        token_type_ids = [0] * (len(question_tokens) + 2) + [1] * (len(word_tokens) + 1)
        bbox = [null_box] * (len(question_tokens) + 2) + word_boxes_expanded + [null_box]
        density_seq = [0.0] * (len(question_tokens) + 2) + word_density_expanded + [0.0]
        attention_mask = [1] * len(input_ids)

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]
            bbox = bbox[:max_length]
            density_seq = density_seq[:max_length]

        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len
            token_type_ids += [0] * pad_len
            bbox += [null_box] * pad_len
            density_seq += [0.0] * pad_len

        context_start = len(question_tokens) + 2
        start_pos, end_pos = find_answer_span(input_ids, answers, context_start, tokenizer)

        batch_ids.append(input_ids)
        batch_mask.append(attention_mask)
        batch_ttids.append(token_type_ids)
        batch_bbox.append(bbox)
        batch_density.append(density_seq)
        batch_start.append(start_pos)
        batch_end.append(end_pos)
        batch_density_groups.append(density_group)
        batch_answers.append(answers)

    return {
        'input_ids': batch_ids,
        'attention_mask': batch_mask,
        'token_type_ids': batch_ttids,
        'bbox': batch_bbox,
        'density_scores': batch_density,
        'start_positions': batch_start,
        'end_positions': batch_end,
        'density_group': batch_density_groups,
        'answers': batch_answers,
    }


def main():
    print("=" * 70)
    print("OFFLINE TOKENIZATION & CACHING")
    print("=" * 70)
    print("Pre-processing datasets to remove CPU bottleneck during training")
    print("=" * 70)

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache directory: {CACHE_DIR}")
    except Exception as e:
        logger.error(f"Failed to create cache directory: {e}")
        return 1

    print("\n[1/6] Loading tokenizer...")
    try:
        tokenizer = LayoutLMTokenizerFast.from_pretrained(MODEL_NAME)
        print(f"  ✓ Tokenizer: {MODEL_NAME}")
        print(f"  ✓ Vocab size: {tokenizer.vocab_size}")
        print(f"  ✓ Max length: {MAX_SEQ_LENGTH}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return 1

    print("\n[2/6] Loading training subset...")
    train_file = None
    if TRAIN_SUBSET_JSONL.exists():
        train_file = TRAIN_SUBSET_JSONL
        print(f"  ✓ Found: {train_file.name}")
    elif TRAIN_SUBSET_JSON.exists():
        train_file = TRAIN_SUBSET_JSON
        print(f"  ✓ Found: {train_file.name}")
    else:
        logger.error("=" * 70)
        logger.error("TRAINING SUBSET NOT FOUND")
        logger.error("=" * 70)
        logger.error(f"Expected files:\n  - {TRAIN_SUBSET_JSONL}\n  - {TRAIN_SUBSET_JSON}")
        logger.error("REQUIRED ACTION: create stratified subset first (e.g. train_models density_subset)")
        logger.error("=" * 70)
        return 1

    try:
        train_dataset = load_dataset('json', data_files=str(train_file))['train']
        print(f"  ✓ Loaded {len(train_dataset):,} training samples")
        required_cols = ['question', 'words', 'boxes', 'answers', 'token_density_scores', 'density_group']
        missing_cols = [col for col in required_cols if col not in train_dataset.column_names]
        if missing_cols:
            logger.error(f"Training dataset missing required columns: {missing_cols}")
            return 1
        # print(train_dataset[0].keys())  # debug
    except Exception as e:
        logger.error(f"Failed to load training dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    print("\n[3/6] Loading validation set...")
    val_file = None
    if VAL_FILE_JSONL.exists():
        val_file = VAL_FILE_JSONL
        print(f"  ✓ Found: {val_file.name}")
    elif VAL_FILE_JSON.exists():
        val_file = VAL_FILE_JSON
        print(f"  ✓ Found: {val_file.name}")
    else:
        logger.error("=" * 70)
        logger.error("VALIDATION FILE NOT FOUND")
        logger.error("REQUIRED ACTION: Run stratified_data_setup.py first!")
        logger.error("=" * 70)
        return 1

    try:
        val_dataset = load_dataset('json', data_files=str(val_file))['train']
        print(f"  ✓ Loaded {len(val_dataset):,} validation samples")
        required_cols = ['question', 'words', 'boxes', 'answers', 'token_density_scores', 'density_group']
        missing_cols = [col for col in required_cols if col not in val_dataset.column_names]
        if missing_cols:
            logger.error(f"Validation dataset missing required columns: {missing_cols}")
            return 1
    except Exception as e:
        logger.error(f"Failed to load validation dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    print("\n[4/6] Preprocessing training dataset...")
    print("  This may take several minutes...")
    def prepare_train_features(examples):
        return prepare_features(examples, tokenizer, MAX_SEQ_LENGTH)

    columns_to_remove = [col for col in train_dataset.column_names if col not in ['density_group', 'answers']]
    print("  Processing in batches of 16...")
    try:
        train_processed = train_dataset.map(
            prepare_train_features,
            batched=True,
            batch_size=16,
            remove_columns=columns_to_remove,
            desc="Tokenizing training data",
            num_proc=1,
        )
        print(f"  ✓ Processed {len(train_processed):,} training samples")
        print(f"  ✓ Features: {list(train_processed.features.keys())}")
        if len(train_processed) == 0:
            raise ValueError("Processed training dataset is empty!")
        sample = train_processed[0]
        required_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'bbox', 'density_scores', 'start_positions', 'end_positions']
        missing_keys = [key for key in required_keys if key not in sample]
        if missing_keys:
            raise ValueError(f"Processed training dataset missing keys: {missing_keys}")
        if len(sample['input_ids']) != MAX_SEQ_LENGTH:
            raise ValueError(f"input_ids length != MAX_SEQ_LENGTH")
    except Exception as e:
        logger.error(f"Failed to preprocess training dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    print("\n[5/6] Preprocessing validation dataset...")
    def prepare_val_features(examples):
        return prepare_features(examples, tokenizer, MAX_SEQ_LENGTH)
    columns_to_remove = [col for col in val_dataset.column_names if col not in ['density_group', 'answers']]
    print("  Processing in batches of 16...")
    try:
        val_processed = val_dataset.map(
            prepare_val_features,
            batched=True,
            batch_size=16,
            remove_columns=columns_to_remove,
            desc="Tokenizing validation data",
            num_proc=1,
        )
        print(f"  ✓ Processed {len(val_processed):,} validation samples")
        if len(val_processed) == 0:
            raise ValueError("Processed validation dataset is empty!")
        sample = val_processed[0]
        required_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'bbox', 'density_scores', 'start_positions', 'end_positions', 'density_group', 'answers']
        missing_keys = [key for key in required_keys if key not in sample]
        if missing_keys:
            raise ValueError(f"Processed validation dataset missing keys: {missing_keys}")
    except Exception as e:
        logger.error(f"Failed to preprocess validation dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    print("\n[6/6] Saving cached datasets to disk...")
    try:
        import shutil
        if TRAIN_CACHE_DIR.exists():
            logger.info(f"Removing existing training cache: {TRAIN_CACHE_DIR}")
            shutil.rmtree(TRAIN_CACHE_DIR)
        if VAL_CACHE_DIR.exists():
            logger.info(f"Removing existing validation cache: {VAL_CACHE_DIR}")
            shutil.rmtree(VAL_CACHE_DIR)
        print(f"  Saving training cache to: {TRAIN_CACHE_DIR}")
        train_processed.save_to_disk(str(TRAIN_CACHE_DIR))
        print(f"  ✓ Training cache saved ({len(train_processed):,} samples)")
        print(f"  Saving validation cache to: {VAL_CACHE_DIR}")
        val_processed.save_to_disk(str(VAL_CACHE_DIR))
        print(f"  ✓ Validation cache saved ({len(val_processed):,} samples)")
    except Exception as e:
        logger.error(f"Failed to save cached datasets: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    print("\nLoading cached datasets to verify...")
    try:
        train_cached = Dataset.load_from_disk(str(TRAIN_CACHE_DIR))
        val_cached = Dataset.load_from_disk(str(VAL_CACHE_DIR))
        print(f"  ✓ Training samples: {len(train_cached):,}")
        print(f"  ✓ Validation samples: {len(val_cached):,}")
        sample = train_cached[0]
        print(f"\n  Training sample features: {list(sample.keys())}")
        print(f"    input_ids length: {len(sample['input_ids'])}")
        assert isinstance(sample['input_ids'], (list, tuple, np.ndarray))
        assert all(isinstance(x, (int, np.integer)) for x in sample['input_ids'][:10])
        assert len(sample['input_ids']) == MAX_SEQ_LENGTH
        assert len(sample['bbox']) == MAX_SEQ_LENGTH and len(sample['bbox'][0]) == 4
        assert len(sample['density_scores']) == MAX_SEQ_LENGTH
        val_sample = val_cached[0]
        assert 'density_group' in val_sample and 'answers' in val_sample
        print("\n  ✓ All verification checks passed!")
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    print("\n" + "=" * 70)
    print("CACHING COMPLETE")
    print("=" * 70)
    print(f"\nCached: {TRAIN_CACHE_DIR} ({len(train_cached):,}), {VAL_CACHE_DIR} ({len(val_cached):,})")
    # TODO: try caching with num_proc>1 on Linux for faster runs
    print("\n✓ Cache is ready for training!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
