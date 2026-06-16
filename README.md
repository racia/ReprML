
## Results Table

| Task | Setting | Seed | Val Acc | Stddev | Mean False Error | FP std | Mean Churn | FP Churn | FN Churn | L2 norm |
|------|---------|-------|---------|--------|-------------------|--------|------------|----------|----------|---------|
| GLUE ||||||||||
| | ALG     | —     | 90,11   | 0,0060 | 0,0989            | 3,61   | 0,042      | _0,32_     | 0,39     | **1,56**    |
|| IMP     | 4     | **90,83**   | _0,0038_ | 0,0917            | 4,52   | _0,039_      | 0,33     | 0,37     | 1,46    |
|| IMP     | 14    | 90,08   | **0,0068** | 0,0992            | **5,25**   | 0,041      | 0,33     | _0,35_     | _1,32_    |
|| IMP     | 24    | _89,56_   | _0,0028_ | **0,1044**            | 3,52   | 0,045      | **0,34**     | 0,37     | 1,55    |
|| NO     | —    | 90,13   | 0,0054 | 0,0987            | 3,84   | 0,042      | 0,332    | 0,372    | 1,53    |
|| DET      | 24     | 89,72   | 0,0042 | 0,1028            | _3,00_   | **0,048**      | **0,34**     | **0,43**     | 1,52    |

### Top-line metrics (Validation accuracy -stddev)

<img width="551" height="331" alt="Validation accuracies (settings)" src="https://github.com/user-attachments/assets/559c6e06-0f71-4016-8a0f-d82fd1bfbd0c" />

The algorithmic noise controlled with fixed seeds setting, IMP - seed 4, scores highest validation accuracy with 0,001 below average stddev (0,0038). The relative high FP stddev (4,52) suggests that fixed seed runs don't yield same (false) predictions across runs. This may be surprising since identical model initializations with same seeds should result in constant predictions on same data distributions. This is further supported by the implementation-level noise controlled (ALG) setting with 0,006 accuracy stddev scoring relative low FP stdv, showing that even with varied seeds initializations, the predictions divergences stay low. As such, this highlights that prediction stability is not guaranteed with fixed random seed initialization alone. The deterministic setting (0,0042 val. acc stdv) with lowest FP stddev (3,0) seems to confirm this observation, requiring both noise types controlled for more stable cross-run predictions.

### Sub-group metrics (Mean False Error, Churn, L2-Norm)

<img width="500" height="300" alt="False Positives standard deviation" src="https://github.com/user-attachments/assets/5af8de47-a410-4fb0-9dc9-f4a6b789d401" />


### Mean Churn and L2-Norm
For the algorithmic noise setting, where implementation-level noise is controlled, l2-norm is highest (1,56), pointing at more nuanced prediction variability, even with stable validation accuracy of 90,11% (0,006 stddev). This is expected for the algorithmic noise from seed variances.

Additionally, the deterministic setting with fixed seed (24), matrix multiplications and torch algortihms, records second highest Mean False Errors (0,103), comparable to IMP - seed24 (-0,0014). Highest mean churn is scored for this all deterministic setting (0,048), with highest sub-group FN-churn 0,43 (also shared highest FP-churn: 0.34) staying close to the according same seed IMP noise setting, though with a bigger difference in the FN Churn rate (+0,06 change). (However, the algorithmic noise comes in-between with FN churn 0,39, resembling the relative high FN-churn in the implementation noise type controlled runs.) This is remarkable, since this may point to a "saturating" effect of a "badly" chosen seed, where a poorly parametrized model, yields unstable cross-run predictions. As such, under deterministic conditions, specific model vulnerabilities may be exposed, here: as cross-run FN rates differences. As such, full determinism is not a guarantee for prediction determinism, but rather a signal for enforced potential model instabilities under bad conditions for specific data sub-groups. 
