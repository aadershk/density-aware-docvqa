# Density-Aware LayoutLM for DocVQA

Thesis project: **Density-Aware LayoutLM** for Document Visual Question Answering (DocVQA). It studies how OCR token density affects performance and adds per-token density embeddings to LayoutLM.

## Research contribution

- **Problem**: EDA shows that performance drops on dense documents (high token density).
- **Approach**: Custom `DensityAwareLayoutLM` that injects local density embeddings via a small projection layer.
- **Evaluation**: Stratified metrics (Sparse / Medium / Dense) and comparison with baseline and SOTA (Impira).

## Project flow

```
Data (DocVQA + OCR)
       │
       ▼  scripts/stratified_data_setup.py
Data/prepared/  (train/val prepared JSON + density_group, token_density_scores)
       │
       ├───────────────────────────────────────────────────────────────┐
       │                                                               │
       ▼  scripts/cache_data.py (optional)                             ▼  src/train_models.py --task density_subset
Data/cached/train, Data/cached/val                                     (creates train_v1.0_subset_25.json + Data/cached/train_subset)
       │                                                               │
       │  Val cache required for training & evaluation                 │  train_subset + val cache required for baseline/density
       ▼                                                               ▼
src/train_models.py --task baseline | density | density_subset
       │
       ▼  src/eval_models.py
outputs/FINAL_THESIS_RESULTS.json  (Untrained, Baseline, Density, SOTA — ANLS by group)
```

**Order to run:**

1. **Data prep**: `scripts/stratified_data_setup.py` → `Data/prepared/` (train/val JSON + JSONL, `density_thresholds.json`).
2. **Val cache** (needed for training and eval): Run `scripts/cache_data.py`. It expects `Data/prepared/val_v1.0_prepared.json` and, for train cache, `Data/prepared/train_v1.0_subset.json` (or `.jsonl`). If you only need the val cache for eval/training, you can create a minimal train subset or run `density_subset` once (see below) and then run `cache_data` after copying `train_v1.0_subset_25.json` to `train_v1.0_subset.json`.
3. **Training**:  
   - `python -m src.train_models --task density_subset` creates the 25% stratified subset (`train_v1.0_subset_25.json`) and **Data/cached/train_subset**, then trains the density model.  
   - `python -m src.train_models --task baseline` or `--task density` then use **Data/cached/train_subset** and **Data/cached/val** (val from step 2).
4. **Evaluation**: `python -m src.eval_models` loads val cache and runs four setups (Untrained, Baseline, Density, SOTA), prints a table and writes **outputs/FINAL_THESIS_RESULTS.json**.

## Project structure

```
├── DocVQA_EDA.ipynb          # EDA: density vs performance, failure modes
├── README.md
├── scripts/
│   ├── stratified_data_setup.py   # OCR + QA → prepared JSON with density groups & token scores
│   └── cache_data.py              # Pre-tokenize to Data/cached/train & val (removes CPU bottleneck)
├── src/
│   ├── train_models.py             # --task baseline | density | density_subset
│   └── eval_models.py              # Four evals → FINAL_THESIS_RESULTS.json
├── Data/                           # Not in repo
│   ├── train_v1.0_withQT.json
│   ├── val_v1.0_withQT.json
│   ├── ocr/                         # OCR JSONs per document
│   ├── prepared/                   # From stratified_data_setup.py
│   └── cached/                     # From cache_data.py & train_models (train_subset)
└── outputs/                        # Checkpoints, metrics, FINAL_THESIS_RESULTS.json
```

## Setup

### Option A: Quick start (pre-prepared Data — recommended for grading)

Use this if you want to run training/evaluation without regenerating data.

1. **Clone the repo**
   ```bash
   git clone https://github.com/aadershk/density-aware-docvqa.git
   cd density-aware-docvqa
   ```

2. **Download the `Data` folder**  
   The repository does not include `Data/` (it is gitignored). Download the pre-prepared Data from:
   - **Google Drive:** [Data folder](https://drive.google.com/drive/folders/1IOyILPb1D-ot77dXlpElSL3UKNWO7W6Z?usp=sharing)  
   The folder contains: `cached/`, `images/`, `ocr/`, `outputs/`, `prepared/`, and optionally `spdocvqa_ocr.tar.gz` (raw OCR archive).

3. **Place `Data` in the project root**  
   After downloading, extract or move the folder so that your project directory looks like:
   ```
   density-aware-docvqa/
   ├── Data/          ← the downloaded folder (with cached/, prepared/, ocr/, etc. inside)
   ├── scripts/
   ├── src/
   ├── DocVQA_EDA.ipynb
   └── README.md
   ```
   You should have `Data/cached/val`, `Data/prepared/`, etc. Then you can run training and evaluation (see below) without running data-prep or caching scripts.

4. **Environment and dependencies**
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # Linux/macOS: source .venv/bin/activate
   pip install torch transformers datasets tqdm numpy
   ```
   (Python 3.8+; PyTorch 2.x, Transformers 4.30+. For grading we use Python 3.13.)

---

### Option B: From scratch (raw DocVQA)

Use this if you have the official DocVQA dataset and want to regenerate everything.

1. **Clone and environment** — same as Option A (clone repo, create venv, activate).

2. **Dependencies**
   ```bash
   pip install torch transformers datasets tqdm numpy
   ```

3. **DocVQA data**  
   Put raw DocVQA JSONs and OCR under `Data/`:
   - `Data/train_v1.0_withQT.json`, `Data/val_v1.0_withQT.json`
   - `Data/ocr/` — one JSON per document (OCR with words/boxes)

4. **Prepare data and cache**
   ```bash
   python scripts/stratified_data_setup.py
   python scripts/cache_data.py
   ```
   (For `cache_data.py` you may need a train subset; see *Project flow* above.)

## Training

```bash
# 25% stratified subset + train density model (creates train_subset cache)
python -m src.train_models --task density_subset

# Baseline (LayoutLM, no density)
python -m src.train_models --task baseline

# Density model (same data as baseline; use after train_subset exists)
python -m src.train_models --task density
```

| Argument | Default | Description |
|----------|---------|--------------|
| `--task` | required | `baseline`, `density`, or `density_subset` |
| `--output_dir` | task-specific | Checkpoint dir |
| `--per_device_train_batch_size` | 2 | Batch size |
| `--gradient_accumulation_steps` | 16 | Effective batch size |
| `--epochs` | 3 | Epochs |
| `--learning_rate` | 3e-5 | LR |
| `--resume_from_checkpoint` | False | For density/density_subset |

Outputs: `outputs/baseline_experiment/` or `outputs/subset_experiment/` (final model, metrics).

## Evaluation

```bash
python -m src.eval_models
```

Runs: **Untrained** (base LayoutLM), **Baseline** (trained, no density), **Density** (trained with density), **SOTA** (Impira LayoutLM-Document-QA). Requires **Data/cached/val** and trained checkpoints for Baseline/Density. Writes **outputs/FINAL_THESIS_RESULTS.json** and prints ANLS (%) by Sparse / Medium / Dense / Overall.

## Metrics

- **ANLS** (Average Normalized Levenshtein Similarity), threshold 0.5
- Results stratified by density group (Sparse / Medium / Dense)

## Key files

| File | Role |
|------|------|
| `scripts/stratified_data_setup.py` | Merge OCR + QA; 33/66 percentiles; per-token density; write prepared JSON/JSONL. |
| `scripts/cache_data.py` | Offline tokenization → Data/cached/train & val for fast loading. |
| `src/train_models.py` | Baseline and DensityAwareLayoutLM training; density_subset builds 25% subset + train_subset cache. |
| `src/eval_models.py` | Runs four evals on val cache (and Impira on raw val); writes FINAL_THESIS_RESULTS.json. |
| `DocVQA_EDA.ipynb` | EDA and density vs performance analysis. |
