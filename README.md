# Bayesian Virtual Surveys (BVS)

Implementation of **Bayesian Virtual Surveys (BVS)**, a probabilistic framework for synthetic population generation and decision-support under uncertainty.

Developed for **consumer acceptance of recycled plastics** under circular economy regulation.

---

## Overview

BVS combines:

- Spanish national survey data (CIS 3391, n=2,254)
- Aggregate international evidence (13 studies, >28,000 respondents)
- Bayesian networks with Dirichlet-Multinomial conjugate fusion

to generate a **synthetic population of 100,000 individuals** and enable **segment-level probabilistic predictions** for variables not observed in the original dataset.

---

## Pipeline

The pipeline follows the structure described in the paper (Martins et al., 2024).

### Core steps

| Script | Description | Output |
|---|---|---|
| `step1.py` | DAG structure learning (MC³) | `dag_edges_3391.csv` |
| `step2.py` | Prior construction from literature | `step2_dirichlet_priors.json` |
| `step3.py` | Bayesian fusion (Dirichlet-Multinomial) | `step3_fused_priors.json` |
| `step4.py` | Synthetic population generation | `synthetic_surveys_S1.csv` |
| `step5.py` | Validation — 3-level framework | `validation_3level.json` |
| `step6.py` | Variable-level diagnostics (TVD) | `tvd_per_variable_fixed.csv` |
| `step7.py` | Held-out Bayesian validation | `level4_heldout_results.csv` |
| `step8.py` | Comparison with baseline models | `comparison_table.csv` |
| `step9.py` | External holdout validation (Ipsos Spain) | `ipsos_validation.json` |

### Supporting modules

| Script | Description |
|---|---|
| `harmonize_categories.py` | Harmonisation of survey scales across studies |
| `schema_extractor.py` | Schema construction from CIS microdata |
| `ingest_pdf.py` | Extraction of prior information from literature |
| `app_gradio.py` | Interactive decision-support interface |

### Sensitivity analysis

| Script | Description | Output |
|---|---|---|
| `run_k_sensitivity_pipeline.sh` | K-bin sensitivity (K in {3, 5, 7}) over the full pipeline | `k_sensitivity_results/resumen.csv` |

---

## Usage

Run the pipeline sequentially:

```bash
python step1.py
python step2.py
python step3.py
python step4.py
python step5.py
python step6.py
python step7.py
python step8.py
python step9.py
```

---

## Interface

```bash
python app_gradio.py
```

---

## Use Case

BVS is designed for **decision-making under uncertainty**, particularly when:

- key behavioural variables are unobserved in national data
- policy decisions must precede measurement
- cross-national evidence needs to be formally integrated

Applications:

- circular economy policy
- sustainable product adoption forecasting
- market segmentation under regulation
- technological forecasting

---

## Interpretation

BVS produces **probabilistic predictions**, not observed ground truth.

Results should be interpreted as conditional acceptance surfaces under uncertainty. Reported distributions represent the posterior predictive distribution of the synthetic population, not point estimates.

---

## Installation

```bash
pip install -r requirements.txt
```

---



---

## License

MIT
