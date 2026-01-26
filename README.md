# Density-Aware LayoutLM for DocVQA

A research project implementing a **Density-Aware LayoutLM** model for Document Visual Question Answering (DocVQA). This work investigates how OCR token density affects model performance and introduces a novel approach to inject local density information into the LayoutLM architecture.

## Research Contribution

The key innovation is the identification of **OCR density** as a critical failure variable in document understanding models. Documents with high token density (dense documents) consistently show degraded performance. This project:

1. **Identifies the problem**: Through EDA, we show that model performance degrades significantly on dense documents
2. **Proposes a solution**: A custom `DensityAwareLayoutLM` that injects per-token local density embeddings
3. **Provides stratified evaluation**: Metrics broken down by density groups (Sparse, Medium, Dense) to measure the effectiveness

## Architecture

The `DensityAwareLayoutLM` extends `LayoutLMForQuestionAnswering` with:
- A **density projection layer** (`nn.Linear(1, hidden_size)`) that converts scalar density scores to embeddings
- **Density embedding injection** before the transformer encoder
- Support for the standard LayoutLM Question Answering task

```python
# Core innovation
density_embeds = self.density_projection(density_scores.unsqueeze(-1))
inputs_embeds = inputs_embeds + density_embeds
```

## Project Structure

```
├── DocVQA_EDA.ipynb          # Exploratory Data Analysis notebook
├── stratified_data_setup.py  # Data preparation with density features
├── train_density_model.py    # Training script with custom model
├── requirements.txt          # Python dependencies
├── Data/                     # Dataset (not in repo - see setup)
│   ├── train_v1.0_withQT.json
│   ├── val_v1.0_withQT.json
│   ├── test_v1.0_withQT.json
│   ├── ocr/                  # OCR JSON files
│   └── prepared/             # Processed data with density features
└── outputs/                  # Model checkpoints and results
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/density-aware-docvqa.git
cd density-aware-docvqa
```

### 2. Create virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download DocVQA Dataset
Download the DocVQA dataset from the [official source](https://www.docvqa.org/) and place it in the `Data/` directory.

### 5. Prepare the data
```bash
python stratified_data_setup.py
```

This will:
- Calculate density percentiles (33rd, 66th) from training data
- Assign density groups (Sparse, Medium, Dense) to each document
- Compute per-token local density scores (neighbors within 50px radius)
- Save prepared data to `Data/prepared/`

## Training

```bash
python train_density_model.py \
    --batch_size 4 \
    --epochs 3 \
    --learning_rate 5e-5 \
    --gradient_checkpointing
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 4 | Batch size for training |
| `--epochs` | 3 | Number of training epochs |
| `--learning_rate` | 5e-5 | Learning rate |
| `--output_dir` | `outputs/density_model` | Output directory |
| `--gradient_checkpointing` | False | Enable to reduce memory usage |
| `--eval_only` | False | Skip training, only evaluate |

### Output
- Model checkpoints saved each epoch
- `stratified_metrics.json` with performance by density group
- Comparison table showing Sparse vs Medium vs Dense performance

## Metrics

The project uses:
- **ANLS** (Average Normalized Levenshtein Similarity) - standard DocVQA metric
- **Exact Match** - strict matching accuracy

Results are stratified by density group to reveal the performance gap on dense documents.

## Key Files

### `stratified_data_setup.py`
- Loads raw DocVQA JSONs and OCR files
- Computes document-level `ocr_density` (total token count)
- Computes per-token `token_density_scores` (local neighborhood density)
- Assigns `density_group` labels based on percentiles
- Uses multiprocessing for efficient computation

### `train_density_model.py`
- Implements `DensityAwareLayoutLM` model class
- Custom `DocVQADataCollator` for subword-density alignment
- `DensityAwareTrainer` for handling custom forward pass
- Stratified evaluation by density group
- ANLS metric implementation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (recommended)

## Citation

If you use this work, please cite:

```bibtex
@misc{density-aware-docvqa,
  title={Density-Aware LayoutLM for Document Visual Question Answering},
  author={Your Name},
  year={2026},
  howpublished={GitHub}
}
```

## License

MIT License
