# AI to Drug Discovery — IRAK4 IC50 Prediction

Predicting the IC50 of candidate compounds against **IRAK4**, a target in
autoimmune and inflammatory disease, from molecular structure alone.

| | |
| --- | --- |
| **Competition** | [제2회 AI 신약개발 경진대회 "JUMP AI 2024"](https://dacon.io/competitions/official/236336/overview/description) |
| **Host** | Ministry of Health and Welfare · Korea Pharmaceutical and Bio-Pharma Manufacturers Association · DACON |
| **Task** | Regression — predict `IC50_nM` from a SMILES string |
| **Data** | 1,952 ChEMBL IRAK4 IC50 records (assay data provided by Daewoong Pharmaceutical) |
| **Metric** | NRMSE (normalised RMSE) |
| **Result** | Top 7% |

---

## The target

IC50 is the concentration of a compound needed to inhibit 50% of a protein's
activity. **Lower is better** — a low IC50 means the compound is potent at small
doses. The task is to predict that number for IRAK4 given only the molecule's
structure.

## Why pIC50 and not IC50

IC50 in nM spans several orders of magnitude, and a raw-scale regression is
dominated by a handful of weak compounds while the potent ones — the only
interesting ones — contribute almost nothing to the loss.

So training happens in **pIC50** space:

```python
pIC50 = -log10(IC50_nM * 1e-9)
```

This compresses the range to roughly 4–10, makes the distribution approximately
normal, and weights potent and weak compounds comparably. Predictions are
converted back to nM for submission:

```python
IC50_nM = 10 ** (9 - pIC50)
```

The log transform is the single most consequential preprocessing decision in the
notebook.

## Features

Three complementary molecular representations, built from SMILES with **RDKit**:

- **Physicochemical descriptors** — molecular weight, LogP, TPSA, H-bond donors
  and acceptors, rotatable bonds, ring counts, and related RDKit descriptors
- **Fingerprints** — Morgan/ECFP circular fingerprints encoding substructure
  presence
- **Graph representation** — the molecule as a graph (atoms as nodes, bonds as
  edges) for the GNN path via `torch_geometric` and **DeepChem**

Outliers are removed by IQR on the target, and extreme IC50 values (>10,000 nM —
essentially inactive compounds) are optionally filtered, since they carry little
signal about potency but distort the fit.

## Models

| Model | Input |
| --- | --- |
| CatBoost | descriptors + fingerprints |
| XGBoost / LightGBM | descriptors + fingerprints |
| **TabNet** (`pytorch-tabnet`) | descriptors + fingerprints |
| GNN (`torch_geometric`) | molecular graph |

Hyperparameters are tuned with **Optuna**, and the final submission blends the
tabular and graph models — they make different errors, which is the point of
keeping both.

## Setup

```bash
git clone https://github.com/paulms77/AI-to-Drug-Discovery.git
cd AI-to-Drug-Discovery

pip install rdkit deepchem torch_geometric torch-scatter \
            pytorch-tabnet catboost optuna polars scikit-learn
```

Download `train.csv` and `test.csv` from the
[competition data page](https://dacon.io/competitions/official/236336/data) and
point the notebook at them:

```python
main_root = '/content/drive/MyDrive/newdrug/'   # adjust to your data directory
train = get_data(main_root + 'train.csv')
```

## Usage

Run [`newdrug.ipynb`](newdrug.ipynb) top to bottom. The notebook covers
descriptor generation, pIC50 conversion, outlier removal, model training with
Optuna search, and submission assembly.

A GPU helps the GNN and TabNet paths but is not required for the
gradient-boosting models.

## Notes

**Chemical validity is not optional.** Descriptors silently return `None` for
molecules RDKit cannot parse, and those propagate into NaN features. Parsing is
checked before featurising rather than after.

**Small data, high variance.** With under 2,000 compounds, cross-validation fold
choice moves the score more than most model changes. Blending across seeds and
folds is doing real work here, not just decoration.
