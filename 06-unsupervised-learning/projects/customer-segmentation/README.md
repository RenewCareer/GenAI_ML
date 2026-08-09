# Project: Customer Segmentation — Wholesale Distributor Clients

## Status
- [x] Reference solution provided ([`customer_segmentation.ipynb`](customer_segmentation.ipynb)) — work through it, then try the extension ideas at the end yourself.

## Problem Statement
A wholesale food distributor wants to segment its ~440 business clients by annual spending
pattern across product categories, so sales and marketing can tailor outreach per segment.

## Data
440 real wholesale distributor clients, loaded directly from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv) —
annual spend across Fresh, Milk, Grocery, Frozen, Detergents/Paper, and Delicatessen, plus
a real `Channel` (Horeca vs. Retail) and `Region` label.

## Approach
1. Log-transformed heavily right-skewed spending data, then scaled it
2. Chose `k` via elbow method + silhouette score (Section 6.1)
3. Visualized segments via PCA (Section 6.2)
4. **Profiled and named each segment** by its dominant spending category — the actual
   sales-team deliverable
5. Cross-checked segments against the dataset's real `Channel` label as a partial sanity
   check

## Results
- Identified several distinct, interpretable client segments.
- Named each by its dominant spending pattern (e.g. "Fresh-heavy, likely Horeca clients")
  rather than leaving them as anonymous cluster IDs.
- Found partial alignment with the real `Channel` label, a useful sanity check on the
  unsupervised result.

## Extend it yourself
- [ ] Try Hierarchical Clustering instead of K-Means — does the dendrogram suggest a
      different natural `k`?
- [ ] Re-run without the log-transform and compare the elbow/silhouette charts
- [ ] Compute each segment's total revenue contribution to find the distributor's most
      critical segment

## Writeup
This project is a reusable template for any spending-based segmentation task: log-transform
→ scale → choose k → visualize via PCA → profile and name each segment. The naming step is
what turns an algorithm's output into something a business can act on.
