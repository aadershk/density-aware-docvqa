"""Merge OCR + QA, add density stratification (33/66 percentiles) and per-token density scores; write HF-ready JSON."""

import os
import json
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from tqdm import tqdm

BASE_DIR    = Path(__file__).parent.parent.resolve()
DATA_DIR    = BASE_DIR / "Data"
OCR_DIR     = DATA_DIR / "ocr"
OUTPUT_DIR  = BASE_DIR / "Data" / "outputs"
PREPARED_DIR = BASE_DIR / "Data" / "prepared"

PREPARED_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

LOCAL_DENSITY_RADIUS_PX = 50   # neighbors within this many px count toward density
NUM_WORKERS = max(1, os.cpu_count() - 1)


def parse_ocr_file(ocr_path: Path) -> Dict[str, Any]:
    """Parse OCR JSON; return tokens, bboxes (0–1000 for LayoutLM), centroids for density."""
    try:
        with open(ocr_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return {'tokens': [], 'bboxes': [], 'width': None, 'height': None, 'centroids': []}

    tokens, bboxes, centroids = [], [], []
    width, height = None, None

    if isinstance(data, dict) and 'recognitionResults' in data:
        for page in data.get('recognitionResults', []):
            width = page.get('width', 1000)
            height = page.get('height', 1000)
            for line in page.get('lines', []):
                for word in line.get('words', []):
                    text = word.get('text', '')
                    if not text.strip():
                        continue
                    tokens.append(text)
                    bbox = word.get('boundingBox', [])
                    if len(bbox) >= 8:
                        xs = [bbox[0], bbox[2], bbox[4], bbox[6]]
                        ys = [bbox[1], bbox[3], bbox[5], bbox[7]]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        cx = (x_min + x_max) / 2
                        cy = (y_min + y_max) / 2
                        centroids.append((cx, cy))
                        x_min_norm = int(1000 * x_min / width) if width else 0
                        y_min_norm = int(1000 * y_min / height) if height else 0
                        x_max_norm = int(1000 * x_max / width) if width else 0
                        y_max_norm = int(1000 * y_max / height) if height else 0
                        bboxes.append([
                            max(0, min(1000, x_min_norm)),
                            max(0, min(1000, y_min_norm)),
                            max(0, min(1000, x_max_norm)),
                            max(0, min(1000, y_max_norm))
                        ])
                    elif len(bbox) >= 4:
                        cx = (bbox[0] + bbox[2]) / 2
                        cy = (bbox[1] + bbox[3]) / 2
                        centroids.append((cx, cy))
                        x_min_norm = int(1000 * bbox[0] / width) if width else 0
                        y_min_norm = int(1000 * bbox[1] / height) if height else 0
                        x_max_norm = int(1000 * bbox[2] / width) if width else 0
                        y_max_norm = int(1000 * bbox[3] / height) if height else 0
                        bboxes.append([
                            max(0, min(1000, x_min_norm)),
                            max(0, min(1000, y_min_norm)),
                            max(0, min(1000, x_max_norm)),
                            max(0, min(1000, y_max_norm))
                        ])
                    else:
                        centroids.append((0, 0))
                        bboxes.append([0, 0, 0, 0])

    return {
        'tokens': tokens,
        'bboxes': bboxes,
        'width': width,
        'height': height,
        'centroids': centroids,
    }


def compute_token_density_scores_vectorized(
    centroids: List[Tuple[float, float]],
    radius: float = LOCAL_DENSITY_RADIUS_PX,
) -> List[float]:
    """Per-token: count neighbors within radius, normalize to 0–1."""
    n = len(centroids)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    coords = np.array(centroids, dtype=np.float32)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    sq_distances = np.sum(diff ** 2, axis=2)
    radius_sq = radius ** 2
    within_radius = sq_distances <= radius_sq
    np.fill_diagonal(within_radius, False)
    neighbor_counts = np.sum(within_radius, axis=1)
    max_neighbors = max(neighbor_counts.max(), 1)
    density_scores = neighbor_counts / max_neighbors
    return density_scores.tolist()


def compute_token_density_for_document(doc: Dict[str, Any]) -> List[float]:
    """Density scores for one doc from its centroids."""
    centroids = doc.get('centroids', [])
    if not centroids:
        return []
    return compute_token_density_scores_vectorized(centroids, LOCAL_DENSITY_RADIUS_PX)


def load_json_dataset(json_path: Path) -> Dict[str, Any]:
    """Load DocVQA JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_ocr_mapping() -> Dict[str, Path]:
    """Basename -> OCR file path."""
    ocr_files = list(OCR_DIR.rglob("*.json"))
    return {f.stem: f for f in ocr_files}


def get_image_basename(image_path: str) -> str:
    return Path(image_path).stem


def process_document_batch(
    image_paths: List[str],
    ocr_mapping: Dict[str, Path],
) -> Dict[str, Dict]:
    """Parse OCR + density for a batch; return image_path -> processed dict."""
    results = {}
    for img_path in image_paths:
        basename = get_image_basename(img_path)
        if basename in ocr_mapping:
            ocr_data = parse_ocr_file(ocr_mapping[basename])
            density_scores = compute_token_density_for_document(ocr_data)
            results[img_path] = {
                'tokens': ocr_data['tokens'],
                'bboxes': ocr_data['bboxes'],
                'width': ocr_data['width'],
                'height': ocr_data['height'],
                'token_density_scores': density_scores,
                'ocr_token_count': len(ocr_data['tokens']),
            }
        else:
            results[img_path] = {
                'tokens': [],
                'bboxes': [],
                'width': None,
                'height': None,
                'token_density_scores': [],
                'ocr_token_count': 0,
            }
    return results


def parallel_process_documents(
    unique_images: List[str],
    ocr_mapping: Dict[str, Path],
    num_workers: int = NUM_WORKERS,
    batch_size: int = 100,
) -> Dict[str, Dict]:
    """Process all docs in parallel; return image_path -> processed data."""
    batches = [unique_images[i:i + batch_size] for i in range(0, len(unique_images), batch_size)]
    all_results = {}
    print(f"Processing {len(unique_images)} documents with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_document_batch, batch, ocr_mapping): i for i, batch in enumerate(batches)}
        for future in tqdm(as_completed(futures), total=len(batches), desc="OCR Processing"):
            try:
                batch_results = future.result()
                all_results.update(batch_results)
            except Exception as e:
                print(f"Warning: Batch processing error: {e}")
    return all_results


def compute_density_percentiles(ocr_densities: List[int]) -> Tuple[float, float]:
    """(p33, p66) from token counts for stratification."""
    densities = [d for d in ocr_densities if d is not None and d > 0]
    p33 = np.percentile(densities, 33)
    p66 = np.percentile(densities, 66)
    return p33, p66


def assign_density_group(ocr_density: int, p33: float, p66: float) -> str:
    """Sparse / Medium / Dense from percentiles."""
    if ocr_density is None or ocr_density <= 0:
        return 'Sparse'
    if ocr_density <= p33:
        return 'Sparse'
    elif ocr_density <= p66:
        return 'Medium'
    return 'Dense'


def prepare_hf_sample(
    sample: Dict[str, Any],
    ocr_data: Dict[str, Any],
    density_group: str,
) -> Dict[str, Any]:
    """One sample in HF-compatible format (words, boxes, density_group, token_density_scores)."""
    answers = sample.get('answers', [])
    primary_answer = answers[0] if answers else ''
    return {
        'id': str(sample.get('questionId', '')),
        'question': sample.get('question', ''),
        'answers': answers,
        'primary_answer': primary_answer,
        'image': sample.get('image', ''),
        'docId': sample.get('docId', ''),
        'question_types': sample.get('question_types', []),
        'words': ocr_data.get('tokens', []),
        'boxes': ocr_data.get('bboxes', []),
        'ocr_density': ocr_data.get('ocr_token_count', 0),
        'density_group': density_group,
        'token_density_scores': ocr_data.get('token_density_scores', []),
        'image_width': ocr_data.get('width'),
        'image_height': ocr_data.get('height'),
    }


def save_hf_dataset(samples: List[Dict], output_path: Path, split_name: str):
    """Write JSON + JSONL for HF load_dataset."""
    output_data = {
        'version': '1.0',
        'split': split_name,
        'num_samples': len(samples),
        'data': samples,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False)
    print(f"Saved {len(samples)} samples to {output_path}")
    jsonl_path = output_path.with_suffix('.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"Saved JSONL version to {jsonl_path}")


def main():
    print("=" * 70)
    print("STRATIFIED DATA SETUP FOR DENSITY-AWARE DOCVQA")
    print("=" * 70)

    print("\n[1/5] Loading original datasets...")
    train_path = DATA_DIR / "train_v1.0_withQT.json"
    val_path = DATA_DIR / "val_v1.0_withQT.json"
    test_path = DATA_DIR / "test_v1.0.json"
    train_data = load_json_dataset(train_path)
    val_data = load_json_dataset(val_path)
    test_data = None
    if test_path.exists():
        test_data = load_json_dataset(test_path)
        print(f"  Test set loaded: {len(test_data.get('data', []))} samples")
    print(f"  Train: {len(train_data['data'])} samples")
    print(f"  Val: {len(val_data['data'])} samples")

    print("\n[2/5] Building OCR mapping...")
    ocr_mapping = build_ocr_mapping()
    print(f"  Found {len(ocr_mapping)} OCR files")
    all_images = set()
    for sample in train_data['data']:
        all_images.add(sample['image'])
    for sample in val_data['data']:
        all_images.add(sample['image'])
    if test_data:
        for sample in test_data['data']:
            all_images.add(sample['image'])
    unique_images = list(all_images)
    print(f"  Unique images to process: {len(unique_images)}")

    print("\n[3/5] Processing OCR and computing token densities...")
    img_ocr_data = parallel_process_documents(unique_images, ocr_mapping)
    # print(len(img_ocr_data), list(img_ocr_data.keys())[:3])  # debug

    print("\n[4/5] Computing density stratification...")
    train_densities = []
    for sample in train_data['data']:
        img = sample['image']
        ocr = img_ocr_data.get(img, {})
        train_densities.append(ocr.get('ocr_token_count', 0))
    p33, p66 = compute_density_percentiles(train_densities)
    print(f"  Density percentiles (from training set):")
    print(f"    33rd: {p33:.1f} tokens, 66th: {p66:.1f} tokens")
    print(f"    Sparse: <= {p33:.0f}, Medium: {p33:.0f}–{p66:.0f}, Dense: > {p66:.0f}")

    print("\n[5/5] Preparing HuggingFace-compatible datasets...")
    train_samples = []
    train_groups = defaultdict(int)
    for sample in tqdm(train_data['data'], desc="Processing train"):
        img = sample['image']
        ocr = img_ocr_data.get(img, {})
        density = ocr.get('ocr_token_count', 0)
        group = assign_density_group(density, p33, p66)
        train_groups[group] += 1
        train_samples.append(prepare_hf_sample(sample, ocr, group))

    val_samples = []
    val_groups = defaultdict(int)
    for sample in tqdm(val_data['data'], desc="Processing val"):
        img = sample['image']
        ocr = img_ocr_data.get(img, {})
        density = ocr.get('ocr_token_count', 0)
        group = assign_density_group(density, p33, p66)
        val_groups[group] += 1
        val_samples.append(prepare_hf_sample(sample, ocr, group))

    test_samples = []
    test_groups = defaultdict(int)
    if test_data:
        for sample in tqdm(test_data['data'], desc="Processing test"):
            img = sample['image']
            ocr = img_ocr_data.get(img, {})
            density = ocr.get('ocr_token_count', 0)
            group = assign_density_group(density, p33, p66)
            test_groups[group] += 1
            test_samples.append(prepare_hf_sample(sample, ocr, group))

    save_hf_dataset(train_samples, PREPARED_DIR / "train_v1.0_prepared.json", "train")
    save_hf_dataset(val_samples, PREPARED_DIR / "val_v1.0_prepared.json", "val")
    if test_samples:
        save_hf_dataset(test_samples, PREPARED_DIR / "test_v1.0_prepared.json", "test")

    print("\n" + "=" * 70)
    print("STRATIFICATION SUMMARY")
    print("=" * 70)
    print("\nTraining Set Distribution:")
    for group in ['Sparse', 'Medium', 'Dense']:
        count = train_groups[group]
        pct = 100 * count / len(train_samples)
        print(f"  {group:8s}: {count:6d} ({pct:5.1f}%)")
    print("\nValidation Set Distribution:")
    for group in ['Sparse', 'Medium', 'Dense']:
        count = val_groups[group]
        pct = 100 * count / len(val_samples)
        print(f"  {group:8s}: {count:6d} ({pct:5.1f}%)")
    if test_samples:
        print("\nTest Set Distribution:")
        for group in ['Sparse', 'Medium', 'Dense']:
            count = test_groups[group]
            pct = 100 * count / len(test_samples)
            print(f"  {group:8s}: {count:6d} ({pct:5.1f}%)")

    thresholds = {
        'p33': float(p33),
        'p66': float(p66),
        'train_distribution': dict(train_groups),
        'val_distribution': dict(val_groups),
        'test_distribution': dict(test_groups) if test_samples else None,
    }
    with open(PREPARED_DIR / "density_thresholds.json", 'w') as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nSaved thresholds to {PREPARED_DIR / 'density_thresholds.json'}")

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput files: {PREPARED_DIR / 'train_v1.0_prepared.json'}, {PREPARED_DIR / 'val_v1.0_prepared.json'}, ...")
    # TODO: optionally stratify val by same percentiles from val set for ablation
    print("\nUsage: load_dataset('json', data_files={'train': 'train_v1.0_prepared.json'})")


if __name__ == "__main__":
    main()
