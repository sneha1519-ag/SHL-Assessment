# Grammar Scoring Engine for Spoken Audio

Predicts a continuous grammar score (0–5) for 45–60 second spoken English responses,
for the SHL Hiring Assessment 2026 competition.

## Results

| Metric | Value |
|---|---|
| Training RMSE | 0.294 |
| Cross-validated RMSE (5-fold × 3 repeats) | 0.548 ± 0.002 |
| Cross-validated Pearson r | 0.90 |
| Baseline RMSE (always predict the mean) | 1.238 |
| Public leaderboard RMSE | 0.468 |

## Approach

1. **Preprocessing** – audio resampled to 16 kHz mono, DC offset removed, peak-normalised.
   Silence is detected and speech is split at pauses into chunks of at most 25 s.
2. **Speech to text** – Whisper `medium.en` (main transcript) and `tiny.en` (second view).
   Large models quietly correct grammar, so the disagreement between the two is used as a
   clarity feature.
3. **Features (about 170 per recording)**
   - *Grammar and syntax (spaCy):* complete-sentence ratio, clauses per sentence, parse depth,
     subject–verb agreement and a/an errors, part-of-speech and dependency mix, tense use
   - *Vocabulary and disfluency:* type–token ratio, word length, fillers, repetitions
   - *Fluency and audio:* speech rate, pause counts and lengths, noise floor, pitch, MFCC statistics
   - *Text representations:* word and part-of-speech n-grams (TF-IDF), word vectors
   - *Optional (GPU):* sentence embeddings and WavLM audio embeddings
4. **Models** – LightGBM, Extra Trees and Ridge on the hand-crafted features; Ridge on n-grams;
   SVR on word vectors. Out-of-fold predictions are blended by a linear model with
   non-negative weights. All pretrained models stay frozen.
5. **Validation** – repeated stratified 5-fold cross-validation; scalers and vectorisers are
   fitted inside each fold to avoid leakage.

## Key findings

- **Zero scores are noisy recordings.** About 5% of labels are 0, a level the rubric does not
  define. These recordings are dominated by background noise and are detected through
  noise-level and transcript-disagreement features.
- **Delivery matters as much as wording.** Audio and fluency features predict the score more
  strongly than transcript features alone, partly because the recogniser repairs some
  grammar errors.
- **Predictions regress to the middle.** Low scores are over-predicted and high scores
  under-predicted.

## Files

| File | Contents |
|---|---|
| `grammar_scoring_engine.ipynb` | Full pipeline, report, charts and evaluation |
| `submission.csv` | Test predictions (`filename`, `label`) |

## How to run

**Google Colab (recommended, GPU):**
1. Upload `shl-hiring-assessment-2026.zip` to Google Drive.
2. Open the notebook in Colab and set Runtime → Change runtime type → T4 GPU.
3. Run this in the first cell, then Runtime → Run all:

```python
from google.colab import drive
drive.mount('/content/drive')
!unzip -q -n "/content/drive/MyDrive/shl-hiring-assessment-2026.zip" -d /content/data
import os, glob
hits = [p for p in glob.glob('/content/data/**/train.csv', recursive=True) if '__MACOSX' not in p]
os.environ["GSE_DATA_DIR"] = os.path.dirname(hits[0])
os.environ["GSE_CACHE_DIR"] = "/content/drive/MyDrive/gse_cache_gpu"
```

**Local (CPU):** place the zip on the Desktop and run all cells. The notebook unpacks it and
switches to smaller CPU speech models (Whisper `base.en` / `tiny.en`), which lowers accuracy.

Transcripts and features are cached, so re-runs skip the slow steps.

## Requirements

Python 3.10+, numpy, pandas, matplotlib, scipy, librosa, scikit-learn, lightgbm, spacy,
transformers, torch, sentence-transformers (optional), sherpa-onnx (CPU fallback).

## Limitations and next steps

- Part of the signal comes from how a recording sounds, so results should be re-checked on
  audio recorded under different conditions.
- Possible improvements: fine-tuning a text model (e.g. DeBERTa) on transcripts, Whisper
  `large-v3`, and a dedicated grammar-error-correction model.
