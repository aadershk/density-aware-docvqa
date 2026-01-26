"""
Density-Aware LayoutLM Training Script for DocVQA
==================================================
Optimized for: NVIDIA GTX 1650 Ti (4GB VRAM) on Windows

Philosophy: "Reliability over Speed" - No OOM crashes allowed.

Features:
- Gradient Checkpointing (memory-efficient backprop)
- FP16 Mixed Precision Training
- Conservative batch settings (batch=1, accum=16)
- Windows-safe data loading (num_workers=0)
- Memory cleanup callbacks
- OOM exception handling with state saving
- Architecture verification before training
- VRAM monitoring

Author: Senior ML Systems Engineer
"""

import argparse
import gc
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Suppress HuggingFace warnings after first import
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message="Some weights of")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from datasets import Dataset, load_dataset
from transformers import (
    LayoutLMTokenizerFast,
    LayoutLMForQuestionAnswering,
    LayoutLMConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration - Windows & Low VRAM Safe Defaults
# ============================================================================

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "Data" / "prepared"
OUTPUT_DIR = BASE_DIR / "outputs" / "density_model"

MODEL_NAME = "microsoft/layoutlm-base-uncased"
MAX_SEQ_LENGTH = 512

# CRITICAL: Safe defaults for 4GB VRAM
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION = 16
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_EPOCHS = 3

# Windows-safe data loading
DATALOADER_NUM_WORKERS = 0  # Windows multiprocessing is unstable
DATALOADER_PIN_MEMORY = True


# ============================================================================
# Utility Functions
# ============================================================================

def get_vram_usage() -> Tuple[float, float]:
    """Get current and max VRAM usage in MB."""
    if torch.cuda.is_available():
        current = torch.cuda.memory_allocated() / 1024 / 1024
        max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
        return current, max_allocated
    return 0.0, 0.0


def clear_memory():
    """Aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_vram_status(prefix: str = ""):
    """Print current VRAM status."""
    current, max_used = get_vram_usage()
    logger.info(f"{prefix}VRAM: {current:.1f}MB (Peak: {max_used:.1f}MB)")


def safe_makedirs(path: Path):
    """Safely create directories (Windows-compatible)."""
    path = Path(path).resolve()
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================================
# ANLS Metric Implementation
# ============================================================================

def normalized_levenshtein_distance(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein distance between two strings.
    Returns value in [0, 1] where 0 = identical, 1 = completely different.
    """
    if len(s1) == 0 and len(s2) == 0:
        return 0.0
    if len(s1) == 0 or len(s2) == 0:
        return 1.0
    
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    
    return dp[m][n] / max(m, n)


def anls_score(prediction: str, ground_truths: List[str], threshold: float = 0.5) -> float:
    """
    Compute ANLS (Average Normalized Levenshtein Similarity) score.
    ANLS = 1 - NL if NL < threshold else 0
    """
    if not ground_truths:
        return 0.0
    
    prediction = prediction.lower().strip()
    max_score = 0.0
    
    for gt in ground_truths:
        gt = gt.lower().strip()
        nl_dist = normalized_levenshtein_distance(prediction, gt)
        score = 1.0 - nl_dist if nl_dist < threshold else 0.0
        max_score = max(max_score, score)
    
    return max_score


def exact_match_score(prediction: str, ground_truths: List[str]) -> float:
    """Compute exact match (any ground truth)."""
    prediction = prediction.lower().strip()
    for gt in ground_truths:
        if prediction == gt.lower().strip():
            return 1.0
    return 0.0


# ============================================================================
# Custom Model: Density-Aware LayoutLM
# ============================================================================

class DensityAwareLayoutLM(LayoutLMForQuestionAnswering):
    """
    LayoutLM with density embedding injection.
    
    The Innovation: Adds local token density information as an additional
    embedding that gets added to the standard token embeddings before
    the transformer encoder.
    
    Memory Optimizations:
    - Supports gradient checkpointing
    - Efficient attention mask handling
    """
    
    def __init__(self, config: LayoutLMConfig):
        super().__init__(config)
        
        # Density projection: scalar density score -> hidden_size embedding
        self.density_projection = nn.Linear(1, config.hidden_size)
        
        # Initialize with small weights to not disrupt pretrained model
        nn.init.normal_(self.density_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.density_projection.bias)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        density_scores: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        """
        Forward pass with optional density score injection.
        
        Args:
            density_scores: Per-token density scores, shape (batch_size, seq_len).
                           Values should be normalized to [0, 1].
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        
        # Determine batch size and seq length
        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        # Default attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        
        # Get base embeddings from LayoutLM's embedding layer
        if inputs_embeds is None:
            inputs_embeds = self.layoutlm.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )
        
        # Inject density embeddings if provided
        if density_scores is not None:
            density_scores = density_scores.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
            density_embeds = self.density_projection(density_scores.unsqueeze(-1))
            inputs_embeds = inputs_embeds + density_embeds
        
        # Create extended attention mask (2D -> 4D)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=inputs_embeds.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(inputs_embeds.dtype).min
        
        # Prepare head mask
        if head_mask is not None:
            head_mask = self.layoutlm.get_head_mask(head_mask, self.config.num_hidden_layers)
        else:
            head_mask = [None] * self.config.num_hidden_layers
        
        # Pass through transformer encoder
        encoder_outputs = self.layoutlm.encoder(
            inputs_embeds,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = encoder_outputs[0]
        
        # QA head
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        
        if not return_dict:
            output = (start_logits, end_logits) + encoder_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# ============================================================================
# Data Processing
# ============================================================================

@dataclass
class DocVQADataCollator:
    """
    Custom data collator for DocVQA with density scores.
    Optimized for memory efficiency on low-VRAM systems.
    """
    tokenizer: PreTrainedTokenizerBase
    max_length: int = MAX_SEQ_LENGTH
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return self._process_batch(features)
    
    def _process_batch(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process a batch of examples."""
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        batch_bbox = []
        batch_density_scores = []
        batch_start_positions = []
        batch_end_positions = []
        batch_metadata = []
        
        for feature in features:
            processed = self._process_single(feature)
            batch_input_ids.append(processed['input_ids'])
            batch_attention_mask.append(processed['attention_mask'])
            batch_token_type_ids.append(processed['token_type_ids'])
            batch_bbox.append(processed['bbox'])
            batch_density_scores.append(processed['density_scores'])
            batch_start_positions.append(processed['start_position'])
            batch_end_positions.append(processed['end_position'])
            batch_metadata.append(processed['metadata'])
        
        return {
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_mask),
            'token_type_ids': torch.stack(batch_token_type_ids),
            'bbox': torch.stack(batch_bbox),
            'density_scores': torch.stack(batch_density_scores),
            'start_positions': torch.tensor(batch_start_positions, dtype=torch.long),
            'end_positions': torch.tensor(batch_end_positions, dtype=torch.long),
            'metadata': batch_metadata,
        }
    
    def _process_single(self, feature: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single example with subword alignment."""
        
        question = feature.get('question', '') or ''
        words = feature.get('words', []) or []
        boxes = feature.get('boxes', []) or []
        answers = feature.get('answers', []) or []
        density_scores = feature.get('token_density_scores', []) or []
        density_group = feature.get('density_group', 'Unknown')
        
        # Handle empty inputs
        if not words:
            words = ['[EMPTY]']
            boxes = [[0, 0, 0, 0]]
            density_scores = [0.0]
        
        # Ensure alignment
        if len(density_scores) < len(words):
            density_scores = list(density_scores) + [0.0] * (len(words) - len(density_scores))
        elif len(density_scores) > len(words):
            density_scores = list(density_scores)[:len(words)]
        
        if len(boxes) < len(words):
            boxes = list(boxes) + [[0, 0, 0, 0]] * (len(words) - len(boxes))
        elif len(boxes) > len(words):
            boxes = list(boxes)[:len(words)]
        
        # Tokenize question
        question_encoding = self.tokenizer(
            question,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        question_tokens = question_encoding['input_ids']
        
        # Limit question length
        max_question_length = 64
        if len(question_tokens) > max_question_length:
            question_tokens = question_tokens[:max_question_length]
        
        # Tokenize words and expand features to subtokens
        word_tokens = []
        word_boxes_expanded = []
        word_density_expanded = []
        
        for word, box, density in zip(words, boxes, density_scores):
            if not word or not isinstance(word, str):
                continue
            
            word_encoding = self.tokenizer(
                word,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            subtokens = word_encoding['input_ids']
            
            if len(subtokens) == 0:
                continue
            
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                box = [0, 0, 0, 0]
            
            word_tokens.extend(subtokens)
            word_boxes_expanded.extend([list(box)] * len(subtokens))
            word_density_expanded.extend([float(density)] * len(subtokens))
        
        # Handle empty case
        if not word_tokens:
            word_tokens = [self.tokenizer.unk_token_id]
            word_boxes_expanded = [[0, 0, 0, 0]]
            word_density_expanded = [0.0]
        
        # Calculate available context space
        special_tokens_count = 3  # [CLS], [SEP], [SEP]
        max_context_length = self.max_length - len(question_tokens) - special_tokens_count
        max_context_length = max(1, max_context_length)
        
        # Truncate context
        if len(word_tokens) > max_context_length:
            word_tokens = word_tokens[:max_context_length]
            word_boxes_expanded = word_boxes_expanded[:max_context_length]
            word_density_expanded = word_density_expanded[:max_context_length]
        
        # Build final sequence: [CLS] question [SEP] context [SEP]
        cls_token = self.tokenizer.cls_token_id
        sep_token = self.tokenizer.sep_token_id
        pad_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        input_ids = [cls_token] + question_tokens + [sep_token] + word_tokens + [sep_token]
        token_type_ids = [0] * (len(question_tokens) + 2) + [1] * (len(word_tokens) + 1)
        
        null_box = [0, 0, 0, 0]
        bbox = [null_box] * (len(question_tokens) + 2) + word_boxes_expanded + [null_box]
        density_seq = [0.0] * (len(question_tokens) + 2) + word_density_expanded + [0.0]
        attention_mask = [1] * len(input_ids)
        
        # Safety truncation
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            token_type_ids = token_type_ids[:self.max_length]
            bbox = bbox[:self.max_length]
            density_seq = density_seq[:self.max_length]
        
        # Padding
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [pad_token] * padding_length
            attention_mask += [0] * padding_length
            token_type_ids += [0] * padding_length
            bbox += [null_box] * padding_length
            density_seq += [0.0] * padding_length
        
        # Find answer span
        context_start = len(question_tokens) + 2
        start_position, end_position = self._find_answer_span(input_ids, answers, context_start)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'bbox': torch.tensor(bbox, dtype=torch.long),
            'density_scores': torch.tensor(density_seq, dtype=torch.float),
            'start_position': start_position,
            'end_position': end_position,
            'metadata': {
                'answers': answers,
                'density_group': density_group,
                'question': question,
            }
        }
    
    def _find_answer_span(
        self, 
        input_ids: List[int], 
        answers: List[str],
        context_start: int
    ) -> Tuple[int, int]:
        """Find answer span in tokenized input."""
        
        if not answers:
            return 0, 0
        
        for answer in answers:
            answer_encoding = self.tokenizer(
                answer,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            answer_tokens = answer_encoding['input_ids']
            
            if len(answer_tokens) == 0:
                continue
            
            for i in range(context_start, len(input_ids) - len(answer_tokens) + 1):
                if input_ids[i:i + len(answer_tokens)] == answer_tokens:
                    return i, i + len(answer_tokens) - 1
        
        return 0, 0


def preprocess_dataset(dataset: Dataset, tokenizer: LayoutLMTokenizerFast) -> Dataset:
    """Preprocess dataset with progress bar."""
    
    def _validate_example(example):
        return {
            'question': example.get('question', ''),
            'words': example.get('words', []),
            'boxes': example.get('boxes', []),
            'answers': example.get('answers', []),
            'token_density_scores': example.get('token_density_scores', []),
            'density_group': example.get('density_group', 'Unknown'),
            'id': example.get('id', ''),
        }
    
    return dataset.map(_validate_example, desc="Validating examples")


# ============================================================================
# Memory Cleanup Callback
# ============================================================================

class MemoryCleanupCallback(TrainerCallback):
    """
    Callback to aggressively clean up memory after each epoch and evaluation.
    Critical for preventing OOM on 4GB VRAM.
    """
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step_count = 0
    
    def on_step_end(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ):
        self.step_count += 1
        
        # Log VRAM every N steps
        if self.step_count % self.log_interval == 0:
            current, max_used = get_vram_usage()
            logger.info(f"Step {state.global_step} | VRAM: {current:.1f}MB / Peak: {max_used:.1f}MB")
        
        return control
    
    def on_epoch_end(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ):
        logger.info("Epoch complete - Running memory cleanup...")
        clear_memory()
        print_vram_status("Post-epoch cleanup | ")
        return control
    
    def on_evaluate(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ):
        logger.info("Evaluation complete - Running memory cleanup...")
        clear_memory()
        print_vram_status("Post-eval cleanup | ")
        return control
    
    def on_save(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ):
        logger.info("Checkpoint saved - Running memory cleanup...")
        clear_memory()
        return control


# ============================================================================
# Custom Trainer with Density Scores
# ============================================================================

class DensityAwareTrainer(Trainer):
    """Trainer that handles density_scores in forward pass."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override to pass density_scores to model."""
        
        # Extract metadata (not used in forward pass)
        inputs.pop('metadata', None)
        
        # Forward pass with density scores
        outputs = model(**inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """Override to handle metadata and density scores."""
        
        inputs.pop('metadata', None)
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            
        if prediction_loss_only:
            return (loss, None, None)
        
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        logits = (start_logits, end_logits)
        labels = (inputs.get('start_positions'), inputs.get('end_positions'))
        
        return (loss, logits, labels)


# ============================================================================
# Architecture Verification
# ============================================================================

def verify_architecture(
    model: DensityAwareLayoutLM,
    tokenizer: LayoutLMTokenizerFast,
    device: torch.device
) -> bool:
    """
    Verify the custom density projection layer is working correctly.
    Passes a single dummy batch through the model.
    """
    logger.info("Running architecture verification...")
    
    try:
        model.eval()
        
        # Create dummy batch
        batch_size = 1
        seq_length = 128
        
        dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length), device=device)
        dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=device)
        dummy_token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long, device=device)
        dummy_bbox = torch.zeros(batch_size, seq_length, 4, dtype=torch.long, device=device)
        dummy_density_scores = torch.rand(batch_size, seq_length, device=device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask,
                token_type_ids=dummy_token_type_ids,
                bbox=dummy_bbox,
                density_scores=dummy_density_scores,
            )
        
        # Verify outputs
        assert outputs.start_logits is not None, "start_logits is None"
        assert outputs.end_logits is not None, "end_logits is None"
        assert outputs.start_logits.shape == (batch_size, seq_length), f"Unexpected start_logits shape: {outputs.start_logits.shape}"
        assert outputs.end_logits.shape == (batch_size, seq_length), f"Unexpected end_logits shape: {outputs.end_logits.shape}"
        
        # Verify density projection is being used
        assert hasattr(model, 'density_projection'), "density_projection layer missing"
        assert model.density_projection.weight.shape == (model.config.hidden_size, 1), "density_projection has wrong shape"
        
        # Test without density scores (fallback behavior)
        with torch.no_grad():
            outputs_no_density = model(
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask,
                token_type_ids=dummy_token_type_ids,
                bbox=dummy_bbox,
                density_scores=None,
            )
        
        assert outputs_no_density.start_logits is not None, "Fallback mode failed"
        
        # Clean up
        del dummy_input_ids, dummy_attention_mask, dummy_token_type_ids, dummy_bbox, dummy_density_scores
        del outputs, outputs_no_density
        clear_memory()
        
        logger.info("=" * 50)
        logger.info("ARCHITECTURE VERIFICATION PASSED")
        logger.info("- Density projection layer: OK")
        logger.info("- Forward pass with density: OK")
        logger.info("- Forward pass without density (fallback): OK")
        logger.info("- Output shapes: OK")
        logger.info("=" * 50)
        
        model.train()
        return True
        
    except Exception as e:
        logger.error(f"Architecture verification FAILED: {e}")
        return False


# ============================================================================
# Evaluation Functions
# ============================================================================

def decode_predictions(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: LayoutLMTokenizerFast,
    max_answer_length: int = 64,
) -> List[str]:
    """Decode model predictions to text."""
    
    predictions = []
    
    for i in range(start_logits.size(0)):
        start_idx = start_logits[i].argmax().item()
        end_idx = end_logits[i].argmax().item()
        
        if end_idx < start_idx:
            end_idx = start_idx
        if end_idx - start_idx > max_answer_length:
            end_idx = start_idx + max_answer_length
        
        tokens = input_ids[i, start_idx:end_idx + 1].tolist()
        prediction = tokenizer.decode(tokens, skip_special_tokens=True)
        predictions.append(prediction.strip())
    
    return predictions


def evaluate_stratified(
    model: DensityAwareLayoutLM,
    eval_dataloader: DataLoader,
    tokenizer: LayoutLMTokenizerFast,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model with stratified metrics by density group.
    Memory-efficient implementation with periodic cleanup.
    """
    
    model.eval()
    
    results_by_group = {
        'Sparse': {'predictions': [], 'ground_truths': []},
        'Medium': {'predictions': [], 'ground_truths': []},
        'Dense': {'predictions': [], 'ground_truths': []},
        'Overall': {'predictions': [], 'ground_truths': []},
    }
    
    eval_progress = tqdm(eval_dataloader, desc="Stratified Evaluation", leave=False)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_progress):
            metadata = batch.pop('metadata')
            batch.pop('start_positions', None)
            batch.pop('end_positions', None)
            
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            outputs = model(**batch)
            
            preds = decode_predictions(
                outputs.start_logits.cpu(),
                outputs.end_logits.cpu(),
                batch['input_ids'].cpu(),
                tokenizer
            )
            
            for pred, meta in zip(preds, metadata):
                group = meta['density_group']
                ground_truths = meta['answers']
                
                if group in results_by_group:
                    results_by_group[group]['predictions'].append(pred)
                    results_by_group[group]['ground_truths'].append(ground_truths)
                
                results_by_group['Overall']['predictions'].append(pred)
                results_by_group['Overall']['ground_truths'].append(ground_truths)
            
            # Periodic memory cleanup
            if batch_idx % 50 == 0:
                clear_memory()
    
    # Compute metrics
    metrics = {}
    
    for group, data in results_by_group.items():
        if len(data['predictions']) == 0:
            metrics[group] = {'anls': 0.0, 'exact_match': 0.0, 'count': 0}
            continue
        
        anls_scores = []
        em_scores = []
        
        for pred, gts in zip(data['predictions'], data['ground_truths']):
            anls_scores.append(anls_score(pred, gts))
            em_scores.append(exact_match_score(pred, gts))
        
        metrics[group] = {
            'anls': np.mean(anls_scores) * 100,
            'exact_match': np.mean(em_scores) * 100,
            'count': len(data['predictions']),
        }
    
    clear_memory()
    return metrics


def print_stratified_results(metrics: Dict[str, Dict[str, float]]):
    """Print stratified evaluation results as a table."""
    
    print("\n" + "=" * 70)
    print("STRATIFIED EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Group':<12} {'Count':>8} {'ANLS':>10} {'Exact Match':>12}")
    print("-" * 70)
    
    for group in ['Sparse', 'Medium', 'Dense', 'Overall']:
        if group in metrics:
            m = metrics[group]
            count = m['count']
            anls = m['anls']
            em = m['exact_match']
            print(f"{group:<12} {count:>8} {anls:>10.2f}% {em:>11.2f}%")
    
    print("=" * 70)
    
    if 'Sparse' in metrics and 'Dense' in metrics:
        sparse_anls = metrics['Sparse']['anls']
        dense_anls = metrics['Dense']['anls']
        gap = sparse_anls - dense_anls
        print(f"\nPerformance Gap (Sparse - Dense): {gap:+.2f}% ANLS")
        if gap > 5:
            print("-> Significant gap: Model struggles on dense documents")
        elif gap < -5:
            print("-> Dense documents perform better (unexpected)")
        else:
            print("-> Balanced performance across density groups")
    
    print()


# ============================================================================
# Safe Training with OOM Handling
# ============================================================================

def safe_train(
    trainer: DensityAwareTrainer,
    output_dir: Path,
    model: DensityAwareLayoutLM,
    tokenizer: LayoutLMTokenizerFast,
) -> bool:
    """
    Wrapper around trainer.train() with OOM exception handling.
    Attempts to save state before exiting on OOM.
    """
    
    try:
        logger.info("Starting training...")
        print_vram_status("Pre-training | ")
        
        trainer.train()
        
        logger.info("Training completed successfully!")
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("=" * 70)
            logger.error("OUT OF MEMORY ERROR DETECTED")
            logger.error("=" * 70)
            logger.error(f"Error: {e}")
            
            # Attempt to save current state
            logger.info("Attempting to save current state...")
            try:
                clear_memory()
                emergency_dir = output_dir / "emergency_checkpoint"
                safe_makedirs(emergency_dir)
                
                model.save_pretrained(emergency_dir)
                tokenizer.save_pretrained(emergency_dir)
                
                logger.info(f"Emergency checkpoint saved to: {emergency_dir}")
                logger.info("You can resume training from this checkpoint.")
                
            except Exception as save_error:
                logger.error(f"Failed to save emergency checkpoint: {save_error}")
            
            logger.error("=" * 70)
            logger.error("RECOMMENDATIONS:")
            logger.error("1. Reduce batch_size (currently using the minimum of 1)")
            logger.error("2. Reduce max_seq_length from 512 to 384 or 256")
            logger.error("3. Enable more aggressive gradient checkpointing")
            logger.error("4. Close other GPU-using applications")
            logger.error("=" * 70)
            
            return False
        else:
            raise  # Re-raise non-OOM errors
    
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        logger.info("Saving current state...")
        
        try:
            interrupt_dir = output_dir / "interrupted_checkpoint"
            safe_makedirs(interrupt_dir)
            model.save_pretrained(interrupt_dir)
            tokenizer.save_pretrained(interrupt_dir)
            logger.info(f"Interrupted checkpoint saved to: {interrupt_dir}")
        except Exception as e:
            logger.error(f"Failed to save interrupted checkpoint: {e}")
        
        return False


# ============================================================================
# Main Training Function
# ============================================================================

def main(args):
    """Main training and evaluation pipeline."""
    
    print("=" * 70)
    print("DENSITY-AWARE LAYOUTLM TRAINING")
    print("Optimized for: NVIDIA GTX 1650 Ti (4GB VRAM) on Windows")
    print("Philosophy: Reliability over Speed")
    print("=" * 70)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
        print(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")
        print_vram_status("Initial | ")
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
    else:
        logger.warning("CUDA not available - training will be slow on CPU!")
    
    # Create output directory
    output_dir = safe_makedirs(Path(args.output_dir))
    logger.info(f"Output directory: {output_dir}")
    
    # -------------------------------------------------------------------------
    # Load tokenizer and model
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading model and tokenizer...")
    
    tokenizer = LayoutLMTokenizerFast.from_pretrained(MODEL_NAME)
    
    config = LayoutLMConfig.from_pretrained(MODEL_NAME)
    
    # Suppress model loading warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DensityAwareLayoutLM.from_pretrained(MODEL_NAME, config=config)
    
    # Enable gradient checkpointing (CRITICAL for 4GB VRAM)
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing: ENABLED")
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Model: {MODEL_NAME}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print_vram_status("After model load | ")
    
    # -------------------------------------------------------------------------
    # Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[2/6] Verifying architecture...")
    
    if not verify_architecture(model, tokenizer, device):
        logger.error("Architecture verification failed! Aborting.")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------------------------
    print("\n[3/6] Loading datasets...")
    
    train_file = DATA_DIR / "train_v1.0_prepared.jsonl"
    val_file = DATA_DIR / "val_v1.0_prepared.jsonl"
    
    if not train_file.exists():
        logger.error(f"Training file not found: {train_file}")
        logger.error("Please run stratified_data_setup.py first!")
        sys.exit(1)
    
    dataset = load_dataset(
        'json',
        data_files={
            'train': str(train_file),
            'validation': str(val_file),
        }
    )
    
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Validation: {len(dataset['validation'])} samples")
    
    # Preprocess with progress bar
    train_dataset = preprocess_dataset(dataset['train'], tokenizer)
    val_dataset = preprocess_dataset(dataset['validation'], tokenizer)
    
    clear_memory()
    
    # -------------------------------------------------------------------------
    # Setup data pipeline
    # -------------------------------------------------------------------------
    print("\n[4/6] Setting up data pipeline...")
    
    data_collator = DocVQADataCollator(
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LENGTH,
    )
    
    print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"  Data workers: {DATALOADER_NUM_WORKERS} (Windows-safe)")
    print(f"  Pin memory: {DATALOADER_PIN_MEMORY}")
    
    # -------------------------------------------------------------------------
    # Training arguments - SAFE DEFAULTS
    # -------------------------------------------------------------------------
    print("\n[5/6] Configuring training...")
    
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),  # Mixed precision
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=DATALOADER_PIN_MEMORY,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch",  # Standard PyTorch AdamW (no bitsandbytes)
        max_grad_norm=1.0,  # Gradient clipping for stability
    )
    
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size (per device): {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  FP16 Mixed Precision: {training_args.fp16}")
    print(f"  Gradient Checkpointing: {training_args.gradient_checkpointing}")
    print(f"  Optimizer: AdamW (torch)")
    
    # -------------------------------------------------------------------------
    # Setup trainer with memory callback
    # -------------------------------------------------------------------------
    print("\n[6/6] Initializing trainer...")
    
    memory_callback = MemoryCleanupCallback(log_interval=args.vram_log_interval)
    
    trainer = DensityAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[memory_callback],
    )
    
    print_vram_status("Pre-training | ")
    
    # -------------------------------------------------------------------------
    # Train with OOM handling
    # -------------------------------------------------------------------------
    if not args.eval_only:
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        
        training_success = safe_train(trainer, output_dir, model, tokenizer)
        
        if training_success:
            # Save final model
            final_model_dir = output_dir / "final_model"
            safe_makedirs(final_model_dir)
            model.save_pretrained(final_model_dir)
            tokenizer.save_pretrained(final_model_dir)
            logger.info(f"Final model saved to: {final_model_dir}")
        else:
            logger.error("Training did not complete successfully")
            if not args.eval_anyway:
                return None
    
    # -------------------------------------------------------------------------
    # Stratified Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING STRATIFIED EVALUATION")
    print("=" * 70)
    
    clear_memory()
    print_vram_status("Pre-evaluation | ")
    
    eval_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=DATALOADER_PIN_MEMORY,
    )
    
    metrics = evaluate_stratified(model, eval_dataloader, tokenizer, device)
    print_stratified_results(metrics)
    
    # Save metrics
    metrics_file = output_dir / "stratified_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_file}")
    
    # Final memory report
    print("\n" + "=" * 70)
    print("FINAL MEMORY REPORT")
    print("=" * 70)
    current, peak = get_vram_usage()
    print(f"Current VRAM: {current:.1f}MB")
    print(f"Peak VRAM: {peak:.1f}MB")
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        print(f"Total VRAM: {total_vram:.1f}MB")
        print(f"Peak Usage: {peak / total_vram * 100:.1f}%")
    
    print("\nTraining pipeline completed successfully!")
    
    return metrics


# ============================================================================
# Entry Point
# ============================================================================

def parse_args():
    """Parse command line arguments with SAFE defaults for 4GB VRAM."""
    
    parser = argparse.ArgumentParser(
        description="Train Density-Aware LayoutLM for DocVQA (Optimized for 4GB VRAM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training hyperparameters - SAFE DEFAULTS
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size per device (keep at 1 for 4GB VRAM)"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps", "-ga",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION,
        help="Gradient accumulation steps (simulates larger batch)"
    )
    
    parser.add_argument(
        "--learning_rate", "-lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for model and logs"
    )
    
    # Evaluation options
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation (skip training)"
    )
    
    parser.add_argument(
        "--eval_anyway",
        action="store_true",
        help="Run evaluation even if training fails"
    )
    
    # Monitoring
    parser.add_argument(
        "--vram_log_interval",
        type=int,
        default=100,
        help="Log VRAM usage every N steps"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Print configuration summary
    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    main(args)
