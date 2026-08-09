# Project: Recommendation Engine — Evaluated Properly, Including Cold Start

## Status
- [x] Reference solution provided ([`recommendation_engine.ipynb`](recommendation_engine.ipynb)) — work through it, then try the extension ideas at the end yourself.

## Problem Statement
Section 6.5 built collaborative, content-based, and hybrid recommenders but never properly
evaluated them offline, and never tested the hardest real case: a brand-new user with
almost no rating history. This project fixes both gaps.

## Data
The same real MovieLens (ml-latest-small) data as Section 6.5 —
[`ratings.csv`](https://raw.githubusercontent.com/sankalpjain99/Movie-recommendation-system/master/ratings.csv) /
[`movies.csv`](https://raw.githubusercontent.com/sankalpjain99/Movie-recommendation-system/master/movies.csv).

## Approach
1. A proper **per-user** train/test split — holding out a fraction of each user's ratings,
   avoiding the "user with zero training data" failure of a naive random row split
2. An SVD collaborative filter trained on training ratings only, evaluated via RMSE against
   truly held-out ratings, compared to a naive "predict the average" baseline
3. **A concrete cold-start simulation**: a brand-new user with one rating, showing why
   collaborative filtering can't meaningfully serve them yet
4. A hybrid recommender that automatically shifts weight from content-based (cold start)
   toward collaborative (established users) as rating history accumulates

## Results
- The SVD collaborative filter meaningfully beat the naive average-rating baseline on
  truly held-out data.
- Demonstrated the cold-start problem concretely rather than just describing it.
- Built a hybrid with an explicit, simple rule for weighting content-based vs.
  collaborative signal based on how much rating history a user has.

## Extend it yourself
- [ ] Simulate the same new user after 15 ratings — does the automatic weight shift, and
      do recommendations change?
- [ ] Compute RMSE separately for sparse-history vs. established users
- [ ] Try `n_components=5` and `n_components=50` for the SVD — is there a sweet spot?

## Writeup
The central lesson: a recommender's hardest, most business-critical case — brand new users
and items — is exactly where pure collaborative filtering fails, and exactly where a
properly weighted hybrid earns its complexity.
