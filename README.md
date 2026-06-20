# Predicting Amazon Best Sellers

**Course:** STA 6933 — Advanced Topics in Statistical Learning  
**Contributors:** Dulce Ximena Cid Sanabria & Marco Alejandro Ortiz  
**Institution:** The University of Texas at San Antonio  

---

## Overview

This project builds and evaluates five predictive models for Amazon Best Seller status using 1.4 million product listings across 248 categories. The outcome (`isBestSeller`) is rare — occurring in only **0.67%** of the cleaned sample — so the inferential goal is **ranking and probability calibration** rather than discrete classification at a fixed threshold.

All models are trained on product-level attributes available at listing time: price, star rating, review volume, and category.

---

## Class Imbalance

![Class Imbalance](figures/class_imbalance.png)

The positive class (Best Seller) accounts for 0.67% of the analytic sample. Inverse-prevalence sample weights are applied throughout to prevent the loss from collapsing toward the majority class.

---

## Exploratory Analysis

### Top 15 Categories by Best Seller Share

![Category Best Seller Share](figures/category_bestseller_share.png)

Best Seller density varies across categories by an order of magnitude. The most populated categories (Girls' Clothing, Boys' Clothing, Toys & Games) are not the same as the highest-density ones — Tools & Home Improvement and Sports & Outdoors lead on share. This motivates keeping the full 248-category resolution rather than collapsing to a coarse grouping.

---

## Models

| Model | Description |
|---|---|
| **Lasso (small)** | L1-penalized logistic regression with 248 fine-grained category dummies |
| **Lasso (big)** | L1-penalized logistic regression with 18 coarse category groups |
| **GAM** | Generalized Additive Model with penalized splines on `log(reviews)` and `log(price)` |
| **Multilevel Logistic** | Hierarchical logistic with big-category fixed effects and small-category random intercepts |
| **SVM (RBF)** | Radial Basis Function Support Vector Machine with Platt-scaled probabilities |

### GAM Partial Effects

![GAM Partial Effects](figures/gam_partial_effects.png)

The estimated smooth $f_1(\log(1 + \text{reviews}))$ rises steeply at low review counts and flattens for high-review products — a clear **diminishing-returns pattern** that a linear-in-log specification cannot represent. The smooth on $\log(\text{price})$ is approximately monotone-decreasing, indicating lower-priced products are more likely to attain Best Seller status.

---

## Results

### ROC and Precision-Recall Curves

![ROC and PR Curves](figures/roc_pr_curves.png)

The dashed horizontal in the PR panel marks the 0.67% test prevalence baseline. All five models dominate it by a wide margin in the low-recall regime where rare-event ranking is operationally useful.

### AUC Summary

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| **Lasso (small)** | **0.8134** | **0.0435** |
| Multilevel | 0.7905 | 0.0298 |
| SVM (RBF) | 0.7663 | 0.0397 |
| GAM | 0.7577 | 0.0407 |
| Lasso (big) | 0.7072 | 0.0235 |

PR-AUC is the primary evaluation metric given the extreme class imbalance. The fine-grained **Lasso (small)** is the recommended model, achieving a **6.5× PR-AUC lift** over the random baseline.

### Operating-Point Metrics (Top-K)

| Model | Top K | Precision | Recall | Lift |
|---|---|---|---|---|
| Lasso (small) | 0.5% | 0.0956 | 0.0720 | 14.4× |
| Lasso (small) | 1.0% | 0.0857 | 0.1289 | 12.9× |
| GAM | 0.5% | 0.0978 | 0.0735 | 14.7× |
| GAM | 1.0% | 0.0815 | 0.1226 | 12.3× |
| SVM (RBF) | 0.5% | 0.0857 | 0.0645 | 12.9× |
| SVM (RBF) | 1.0% | 0.0705 | 0.1060 | 10.6× |
| Multilevel | 1.0% | 0.0499 | 0.0751 | 7.5× |
| Lasso (big) | 1.0% | 0.0710 | 0.1068 | 10.7× |

At the **top 1% slice**, the Lasso (small) captures 12.9% of all Best Sellers in the catalog with a precision lift of 12.9× over random — directly useful for assortment curation or promotional targeting.

---

## Key Findings

- **Reputation dominates price.** Review volume and star rating are stronger drivers of Best Seller status than price; stars carry the larger coefficient in the linear-additive fit.
- **Non-linearity in reviews matters.** The GAM recovers a diminishing-returns curve in `f(log_reviews)` that a linear model misses.
- **Fine-grained categories carry signal.** Collapsing 248 small categories to 18 coarse groups costs measurable PR-AUC. The random intercept in the multilevel model partially pools this signal away.
- **Numeric interactions are small.** The RBF SVM does not improve over the additive GAM, suggesting non-linear interaction between `log_price`, `log_reviews`, and `stars` is limited.

---

## R Libraries

The project uses 12 R packages spanning data wrangling, machine learning, statistical modeling, and visualization. No deep learning frameworks are used — all five models are classical statistical or kernel-based methods.

### 🤖 Machine Learning

| Package | Role in this project |
|---|---|
| **glmnet** | Fits the two Logistic Lasso models (Models 1a and 1b) via coordinate descent. Handles the sparse 248-category one-hot matrix and selects the regularization strength `λ` by 10-fold cross-validation. This is the core ML workhorse of the project. |
| **e1071** | Fits the RBF Support Vector Machine (Model 4) via `svm()`. Wraps LIBSVM under the hood and provides Platt-scaled probability outputs used for rare-event ranking. |
| **mgcv** | Fits the Generalized Additive Model (Model 2) with penalized regression splines on `log(reviews)` and `log(price)`. Uses REML to select smoothing penalties automatically. Sits at the boundary of statistical modeling and ML — fully non-parametric in its smooth terms. |
| **pROC** | Computes ROC curves and AUC for all five models on the held-out test set. |
| **PRROC** | Computes Precision-Recall curves and PR-AUC — the primary evaluation metric under extreme class imbalance (0.67% prevalence). |

### 📐 Statistical Modeling

| Package | Role in this project |
|---|---|
| **lme4** | Fits the Multilevel (Hierarchical) Logistic Regression (Model 3) via `glmer()`. Estimates fixed effects on 18 coarse category groups and random intercepts on 248 fine-grained categories using the BOBYQA optimizer. |
| **broom.mixed** | Tidies the `glmer` output from `lme4` into a clean data frame of fixed-effect estimates, standard errors, and p-values for display in the report. |
| **Matrix** | Provides sparse matrix support (`sparse.model.matrix`) for efficiently encoding the 248-category one-hot design matrix fed into `glmnet`. Without sparsity, storing the full dense matrix for 1.2M rows would be prohibitive. |

### 🔧 Data Wrangling & Utilities

| Package | Role in this project |
|---|---|
| **tidyverse** | Umbrella package covering `dplyr` (data manipulation), `tidyr` (reshaping), `readr` (CSV import), `purrr` (functional iteration in the operating-point metrics function), and `ggplot2` (all visualizations). |
| **scales** | Formats axis labels as percentages and comma-separated numbers in `ggplot2` charts, and in the summary tables. |
| **knitr** | Powers the R Markdown rendering pipeline — executes each code chunk and weaves output into the PDF report. |
| **gridExtra** | Arranges multiple `ggplot2` panels side-by-side (e.g., the ROC and PR curve plots displayed together). |

---

## Repository Structure

```
amazon-bestseller-prediction/
├── data/
│   ├── amazon_products.csv       # 1.4M product listings (see note below)
│   └── amazon_categories.csv     # Category ID lookup table
├── figures/
│   ├── class_imbalance.png
│   ├── category_bestseller_share.png
│   ├── roc_pr_curves.png
│   └── gam_partial_effects.png
├── report/
│   ├── milestone4_v3.Rmd         # Full analysis in R Markdown
│   └── milestone4_v3.pdf         # Knitted PDF report
└── README.md
```

> **Data note:** `amazon_products.csv` may exceed GitHub's 100 MB file size limit.

---

## Reproducing the Analysis

**Requirements:** R ≥ 4.2 with the following packages:

```r
install.packages(c(
  "tidyverse", "glmnet", "mgcv", "pROC", "PRROC",
  "Matrix", "scales", "knitr", "gridExtra",
  "e1071", "lme4", "broom.mixed"
))
```

> **Runtime note:** The full pipeline — especially the fine-grained Lasso (10-fold CV) and GAM on 887K rows — takes approximately 30–60 minutes depending on hardware.

---

## Methods Summary

- **Pre-processing:** log-transformed `price` and `reviews`; filtered zero-price and unrated listings (retains 89% of raw data); joined category labels; constructed 18 coarse `big_category` groups via regex rules.
- **Class imbalance:** inverse-prevalence sample weights throughout; evaluation uses PR-AUC and operating-point metrics rather than accuracy.
- **Model selection:** 10-fold CV on ROC-AUC for both Lassos; REML for GAM spline penalties; BOBYQA optimizer for the multilevel logistic; stratified subsampling for the SVM (all positives + 5:1 negatives).
- **Evaluation:** PR-AUC, ROC-AUC, and top-K metrics at K = 0.5%, 1%, 5%, 10%.

---

## Tech Stack

`R` · `tidyverse` · `glmnet` · `mgcv` · `lme4` · `e1071` · `pROC` · `PRROC` · `Matrix` · `broom.mixed` · `scales` · `gridExtra`
