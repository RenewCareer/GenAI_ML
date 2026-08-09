# Project: Fraud Detection — An Investigation-Budget-Aware Anomaly System

## Status
- [x] Reference solution provided ([`fraud_detection_unsupervised.ipynb`](fraud_detection_unsupervised.ipynb)) — work through it, then try the extension ideas at the end yourself.

## Problem Statement
A card issuer's fraud team can only manually review a limited number of flagged
transactions per day. This project reframes Section 6.3's anomaly detection around that
real constraint: given a fixed daily review budget, how should transactions be ranked to
catch the most fraud?

## Data
The same real 284,807-transaction dataset as Section 6.3, loaded from
[`nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection`](https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv).

## Approach
1. PCA visualization of legitimate vs. fraud across the full dataset
2. A stratified working sample (all fraud + a legitimate sample), needed because LOF
   doesn't scale to 284,807 rows
3. A 2-method ensemble (Isolation Forest + LOF), averaged after normalizing each score
4. **Precision@k and recall@k** at several realistic daily review budgets — the actual
   business question, not just ROC AUC
5. An honest look at which fraud the ensemble systematically misses

## Results
- The ensemble outperforms either individual method on ROC AUC.
- Translated performance into a concrete staffing conversation: "reviewing the top N
  transactions catches X% of fraud."
- Found a measurable difference between caught and missed fraud by transaction amount —
  a concrete, investigable gap rather than an unexamined aggregate number.

## Extend it yourself
- [ ] Add One-Class SVM as a third ensemble member
- [ ] Weight the ensemble toward whichever method has the higher standalone AUC
- [ ] Compare Isolation Forest alone against the ensemble on precision@k directly

## Writeup
The real contribution here isn't a new algorithm — it's translating unsupervised anomaly
scores into the operational question a fraud team actually faces: given a fixed number of
investigators, which transactions get reviewed first?
