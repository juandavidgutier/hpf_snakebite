# hfp_snakebite

**Human Footprint Effect on Snakebite Risk and Heterogeneous Effect of Rural Poverty**

**Authors**: Juan David Gutiérrez¹, Carlos Bravo-Vega²

¹ Instituto Masira, Facultad de Medicina y Ciencias de la Salud, Universidad de Santander, Bucaramanga, Colombia
² Grupo de investigación en Biología Matemática y Computacional (BIOMAC), Departamento de Ingeniería Biomédica, Universidad de los Andes, Bogotá, Colombia

---

## Overview

This repository contains the public analysis pipeline for a study examining the association between the rural Human Footprint (HFP) — an index of cumulative human pressure on the landscape — and excess snakebite risk across Colombian municipalities from 2007 to 2023. The analysis proceeds in two stages:

1. **Excess burden estimation**: A spatio-temporal model fitted with Integrated Nested Laplace Approximations (INLA) using a Besag-York-Mollié (BYM) specification with a first-order random walk (RW1) temporal term and a space-time interaction. This model estimates municipality-year Standardized Incidence Ratios (SIR) and identifies municipalities with excess snakebite risk (posterior probability that SIR > 1).

2. **Causal effect estimation**: Double Machine Learning (DML) via DoWhy + EconML, with a Directed Acyclic Graph (DAG) specifying the assumed causal structure. The treatment is rural HFP, the outcome is excess snakebite risk (binary), and effect modifiers include the Rural Multidimensional Poverty Index (MPI), municipality and year. Sensitivity analyses include refutation tests (random common cause, data subset, placebo treatment, bootstrap) and non-parametric partial R² benchmarking against MPI (Chernozhukov et al., 2021). Cluster bootstrap (resampling entire municipalities) provides cluster-robust standard errors.

This dataset has been processed to ensure complete anonymization and contains no personally identifiable information (PII). All data has been:
⦁	Aggregated at appropriate spatial/temporal scales
⦁	Stripped of any individual identifiers
⦁	Processed to remove direct or indirect identifying elements
The dataset is suitable for public sharing and complies with data privacy standards.
This repository contains datasets that have been carefully processed to protect individual privacy:
What is NOT included:
⦁	Names, addresses, or contact information
⦁	Individual-level identifiers
⦁	Any data that could be used to re-identify individuals

**Study period**: 2007–2023 (17 years)
**Geographic scope**: 1,005 municipalities in Colombia
**Software**: R v4.2.1, INLA, Python 3, DoWhy, EconML, XGBoost, scikit-learn



---

## Repository Structure

```
hfp_snakebite/
├── data.csv                         # Public analysis dataset (municipality-year panel)
├── excess_INLA.R                    # Stage 1: Spatio-temporal INLA model for excess SIR
├── hfp_snakebite_estimation.py      # Stage 2: DML causal effect estimation + CATE + sensitivity
├── model_selection.py               # Model selection: R-Score comparison across DML configurations
│
├── map/                             # Geographic data (shapefile — reference only)
│   └── Col.shp, Col.shx, Col.dbf, etc.
│
└── README.md                        # This file
```

---

## Setup

### 1. Configure paths

The scripts contain hard-coded local filepaths (`D:/...`). Before running, update the `read.csv()` path in each script to point to the local copy of `data.csv` in this repository.

### 2. Install R packages

```r
install.packages(c(
  "dplyr", "sf", "INLA", "spdep"
))
```

> **Note:** INLA is not on CRAN. Install from https://www.r-inla.org/download:
> ```r
> install.packages("INLA", repos = c(INLA = "https://inla.r-inla-download.org/R/stable"), dep = TRUE)
> ```

### 3. Install Python packages

```bash
pip install pandas numpy scikit-learn xgboost econml dowhy matplotlib joblib
```

---

## Public Dataset

The primary analysis dataset is **`data.csv`**. Municipality identifiers (DANE codes — the standard geographic classification codes assigned by the Colombian National Administrative Department of Statistics) are retained at the municipality level (no individual-level data). Geographic identifiers below municipality resolution have been removed.

### Variables

| Variable               | Description |
|------------------------|-------------|
| `DANE`                 | Municipality code (DANE) |
| `Year`                 | Calendar year |
| `DANE_Year`            | Municipality-year identifier (e.g., `5001-2007`) |
| `period`               | Sequential period number (1–17) |
| `Temperature`          | Mean annual temperature (°C) |
| `Rainfall`             | Mean annual rainfall (mm) |
| `EVI`                  | Enhanced Vegetation Index |
| `Forest`               | Forest cover (%) |
| `Deforest`             | Deforestation (%) |
| `Fire`                 | Fire activity (%) |
| `Mining`               | Mining activity (%) |
| `Coca`                 | Coca cultivation area (%) |
| `MPI_rural`            | Rural Multidimensional Poverty Index (%) |
| `MPI_urban`            | Urban Multidimensional Poverty Index (%) |
| `HFP_rural`            | Rural Human Footprint index |
| `HFP_urban`            | Urban Human Footprint index |
| `cases_rural`          | Observed rural snakebite cases |
| `exp_rural`            | Expected rural snakebite cases |
| `cases_urban`          | Observed urban snakebite cases |
| `exp_urban`            | Expected urban snakebite cases |
| `SIR_mean`             | Posterior mean Standardized Incidence Ratio |
| `SIR_lwr95`            | 2.5% posterior quantile of SIR |
| `SIR_upr95`            | 97.5% posterior quantile of SIR |
| `excess`               | Binary indicator: 1 if SIR_lwr95 > 1 (excess risk) |

---

## Running the Pipeline

### Stage 1: Excess snakebite risk estimation (INLA)

```bash
Rscript excess_INLA.R
```

This script:
- Reads the municipality shapefile from `map/Col.shp`
- Builds a spatial adjacency matrix for the BYM model
- Fits a spatio-temporal model (Besag + IID + RW1 + space-time interaction) with negative binomial likelihood
- Computes posterior SIR and identifies municipalities with excess risk
- Saves results back to `data.csv`

> **Note:** Requires the INLA R package and the shapefile in `map/`.

### Stage 2: Causal effect estimation (DML)

```bash
python hfp_snakebite_estimation.py
```

This script:
- Loads and standardizes the data
- Encodes municipality and municipality-year identifiers
- Constructs lagged covariates (deforestation and fire at t−1)
- Specifies a DAG and identifies the causal estimand via DoWhy
- Fits a DML model with XGBoost nuisance models and LassoCV final model
- Computes ATE and CATE (conditional on Rural MPI)
- Runs refutation tests: random common cause, data subset, placebo treatment, bootstrap
- Conducts non-parametric partial R² sensitivity analysis (Chernozhukov et al., 2021)
- Runs a cluster bootstrap (50 iterations) for cluster-robust standard errors
- Generates a CATE plot showing how the HFP effect varies with Rural MPI

### Model selection

```bash
python model_selection.py
```

This script compares 10 DML model configurations (varying XGBoost `n_estimators` and `max_depth`) using the R-Score criterion to select the best-performing nuisance models.

---

## Results Summary

| Metric             | Estimate (ATE) | 95% CI           |
|--------------------|----------------|------------------|
| Main specification | −0.117         | −0.154 to −0.103 |

A one-standard-deviation increase in rural HFP was associated with an 11.7 percentage-point lower probability of excess snakebite cases. The negative association was weaker in municipalities with higher rural poverty (MPI), suggesting that poorer communities benefit less from landscape transformation and remain disproportionately vulnerable.

---

## Citation

If you use this repository or its data, please cite:

> Gutiérrez, J.D., & Bravo-Vega, C. Human Footprint Effect on Snakebite Risk and Heterogeneous Effect of Rural Poverty
