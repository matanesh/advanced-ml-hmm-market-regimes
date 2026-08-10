# ARCHIVED QUICK COMPARISON — EXPLORATORY ONLY

> This quick test did not use the final canonical protocol and is not evidence for the submission conclusions. It is preserved for provenance. Use `reports/report.pdf` and `experiments_canonical/canonical_20260810_014841/` as the authoritative sources.

## Historical Summary of Model Comparison (Quick Test: 2020-01-01 to present)

We compared the following models on five assets (SPY, NVDA, DIS, ADBE, QQQ):
- Gaussian HMM with K = 2,3,4 and covariance types 'full' and 'diag'
- Logistic Regression
- Random Forest

**Key Findings:**

1. **Logistic Regression** often achieved the highest Directional Prediction Accuracy (DPA) among the tested models:
   - SPY: 0.5589
   - QQQ: 0.5671
   - ADBE: 0.5061 (close to best HMM)
   - DIS: 0.5264
   - NVDA: 0.5346 (tied with several HMMs)

2. **HMM Performance**:
   - For SPY, HMM K=3 (full or diag) reached ~0.55 DPA, slightly below Logistic Regression.
   - For NVDA, HMM K=4 full matched Logistic Regression at ~0.5467, while other HMMs were lower.
   - For DIS, HMM K=4 full/diag gave ~0.5163, below Logistic Regression.
   - For ADBE, HMM K=3 full gave ~0.5264, above Logistic Regression (0.5061).
   - For QQQ, HMM K=4 diag gave ~0.5407, below Logistic Regression.

3. **Random Forest** performance was variable, sometimes competitive but often lower than Logistic Regression and HMM.

4. **Covariance Type**: 'diag' sometimes performed similarly to 'full', but not consistently better.

**Interpretation:**
- The naive baseline (predicting the training mean return) is a strong benchmark; however, in this recent period, Logistic Regression often outperforms it and HMM.
- HMM can still capture useful regime structure, as seen in our earlier economic interpretation, but for pure direction prediction, simple discriminative models like Logistic Regression may be sufficient.
- The earlier extended experiments (2004-present) showed that HMM occasionally edges out the naive baseline, but the improvement is often small and not always statistically significant.

**Note:** We did not implement a Reinforcement Learning baseline due to time constraints, but the comparison with simple ML classifiers provides a benchmark. If RL is desired, one could frame the problem as a contextual bandit or use Q-learning on discretized states (e.g., HMM states) to learn a policy for predicting direction.

**Files:** Individual comparison results are saved in `model_comparison/{ticker}_model_comparison.csv`.

Next steps if more time is available:
1. Implement a simple RL agent (e.g., Q-learning) that uses HMM-predicted states as features.
2. Test additional HMM variants (e.g., different emission distributions).
3. Perform statistical significance testing (e.g., McNemar's test) on the pairwise comparisons.
4. Extend the analysis to more assets and different time periods.