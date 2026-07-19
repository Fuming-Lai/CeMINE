# CeMINE
LLM-assisted mining of literature-derived CeO₂ synthesis records for nanocrystal morphology prediction, with machine-learning models and experimental validation.

<img width="5420" height="4597" alt="TOC graphic" src="https://github.com/user-attachments/assets/3f063488-4c34-459d-804d-489112054067" />


## Contents

- `ceo2_llm_pdf_input_extractor.py`: Extracts synthesis conditions, morphologies, and exposed facets from PDF papers using an LLM with an OpenAI-compatible Chat Completions API.
- `main.py`: Compares MLR, KNN, MLP, XGBoost, and GOOFS models.
- `Comparison_ML/`: Data processing, baseline models, and evaluation metrics.
- `GOOFS/`: Group-based out-of-fold stacking model.

## Installation

Python 3.10 or later is recommended.

```bash
pip install numpy pandas scikit-learn requests PyMuPDF xgboost lightgbm
```

## Usage

### 1. Extract data from PDF papers

```bash
python ceo2_llm_pdf_input_extractor.py \
  --pdf_dir path/to/pdfs \
  --output_dir extraction_output \
  --api_base http://localhost:11434/v1 \
  --api_key ollama \
  --model your-model-name
```

### 2. Compare the models

Prepare `CeO2_training_set.csv`, then run:

```bash
python main.py \
  --input_csv CeO2_training_set.csv \
  --output_dir Comparison_ML_output \
  --group_col paper_id \
  --random_state 42
```

The input data must contain the target column `target_morphology`, the paper identifier `paper_id`, and the required synthesis feature columns. Every sample must have a valid `paper_id`. Otherwise, the program stops to prevent data leakage between the training and test sets.

## License

The source code is released under the [MIT License](LICENSE).
