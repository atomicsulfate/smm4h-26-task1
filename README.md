# SMM4H 2026 — Task 1: Multilingual ADE Detection (Notebook)

Single-notebook solution for **SMM4H 2026 Task 1**: Detection of Adverse Drug Events (ADE) in multilingual and multi-platform social media posts.

This project focuses on:

- Multilingual generalization (**EN/DE**)
- Cross-lingual transfer / zero-shot evaluation (**FR**)
- Machine translation augmentation (**EN→FR**) and comparison vs zero-shot

## Repository layout

- `smm4h2026_task1_multilingual_ade.ipynb` — main notebook (all logic lives here)
- `environment.yml` — conda/mamba environment specification
- `dataset/`
  - `train_data_SMM4H_2026_Task_1.csv`
  - `dev_data_SMM4H_2026_Task_1.csv` (treated as **test** in the notebook)
- `artifacts/` — cached translations, saved model runs/checkpoints

CSV schema: `id,text,label,origin,type,language,split`

Note: `label` is binary but stored as floats in the CSV (`0.0`, `1.0`).

## Setup

### 1) Create the environment (recommended: mamba)

From the repository root:

```bash
mamba env create -f environment.yml
mamba activate smm4h2026
```

If you don’t have `mamba`, you can use `conda`:

```bash
conda env create -f environment.yml
conda activate smm4h2026
```

The environment includes PyTorch + Hugging Face tooling (`transformers`, `datasets`, `accelerate`, `evaluate`), plus common data/plotting libs.

### 2) Start Jupyter

```bash
jupyter lab
```

Open `smm4h2026_task1_multilingual_ade.ipynb` and run the notebook.

## What the notebook does

### Data protocol

- Uses **English + German** for training.
- Treats the provided `dataset/dev_data_SMM4H_2026_Task_1.csv` as **test**.
- Creates a new **validation** split by sampling from the training file.
  - Validation sampling is stratified by `(language, label)` for EN/DE experiments.

### Modeling setups

All experiments are designed to share the same Transformer backbone and training code (via Hugging Face `Trainer`):

1. **Monolingual**
   - Train EN → evaluate EN test
   - Train DE → evaluate DE test

2. **Multilingual**
   - Train EN+DE → evaluate EN + DE test
   - Zero-shot evaluate on FR test

3. **Machine translation–based**
   - Translate **3000 EN training examples → FR** (stratified by label)
   - Train on the FR translations → evaluate on FR test
   - Compare against zero-shot FR performance from EN-only and EN+DE models

### Default model backbone

- Default: `xlm-roberta-base`
- Swappable (one-line change in notebook), e.g.:
  - `microsoft/mdeberta-v3-base`
  - `microsoft/mdeberta-v3-large`

## Outputs / artifacts

The notebook writes intermediate and final artifacts so reruns are fast and reproducible:

- Translations cache: `artifacts/translations_en_fr_3000_seed42.csv`
- Model runs/checkpoints: `artifacts/runs/` (e.g. `monolingual_en/`, `monolingual_de/`, `multilingual_en_de/`, `translation_trained_fr/`)

If you want a clean rerun, you can delete `artifacts/` (you’ll regenerate translations and retrain models).

## Reproducibility notes

- Splits and sampling use fixed random seeds (see notebook).
- Metrics are reported per language (F1 as primary, plus precision/recall/accuracy) along with confusion matrices.

## Troubleshooting

- If model downloads fail: ensure you have internet access (Hugging Face models are downloaded on first run).
- If you switch to a model that requires SentencePiece: `sentencepiece` is already included in `environment.yml`.
- GPU is optional. If you have CUDA available, PyTorch will use it automatically; otherwise training runs on CPU (slower).
