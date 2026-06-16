## Top-line metrics (Val acc (stddev))

<img width="551" height="331" alt="Validation accuracies (settings)" src="https://github.com/user-attachments/assets/559c6e06-0f71-4016-8a0f-d82fd1bfbd0c" />

## Sub-group metrics (Mean False Error, Churn, L2-Norm)

<img width="500" height="300" alt="False Positives standard deviation" src="https://github.com/user-attachments/assets/5af8de47-a410-4fb0-9dc9-f4a6b789d401" />


### Validation Accuracy (Stddev)
Findings: 1. Validation accuracy is highest (90,83%) for the implementation noise setting with fixed seed 4. Second comes again the algorithmic noise fixed setting (seed 14) with -0,75 acc. percentage points. The varying seeds setting over 10 runs (ALG) closely follows (-0,03 p.p.), and the runs with fixed seed 24 come last (89,56%).
	- 

### False Error Rate and Stability
The mean false error for IMP - seed 4 is lowest (0,0917), followed by fixed seed 14 of 0,0992 (+0,0075)   

### Mean Churn and L2-Norm
For the alg. noise settings the l2-norm is highest (1,56) , pointing at more nuanced prediction variability, even with stable validation accuracies across runs of 90,11% (0,006 stddev). This is expected for the algorithmic noise from seed variances.

For implementation-level noise, the seed 4 configuration performs best (90,83 acc)  with relative high FP std. dev (4,52). This unexpectedly shows that with fixed algorithmic seeds, false predictions may vary across runs, highlighting that identical model parametrization not necessarily results in consistent predictions on identical data distributions. This further shows, that fixed model initialization alone does not guarantee prediction stability. The deterministic setting with lowest FP std. dev (3,00) seems to confirm this, requiring both controlled noise types to ensure low run-to-run predictive divergences.

Additionally, the deterministic setting with fixed seed (24), matrix multiplications and torch algortihms, records second highest Mean False Errors (0,103), comparable to IMP - seed24 (-0,0014). Highest mean churn is scored for this all deterministic setting (0,048), with sub-group FN-churn 0,43 (also shared highest FP-churn: 0.34) staying close to the according same seed IMP noise setting, with a bigger difference in the FN Churn rate (+0,06 change). (However, the algorithmic noise comes in-between with FN churn 0,39, resembling the relative high FN-churn in the implementation noise type controlled runs.) Tis is remarkable, since this may point to a "saturating" effect of a "bad" chosen seed, where a badly parametrized model, yields unstable cross-run predictions. As such, under deterministic conditions, specific model vulnerabilities may be exposed, here: as cross-run FN's divergences. As such, full determinism is not a guarantee for prediction determinism, but rather a signal for enforced potential model instabilities under bad conditions for specific data sub-groups. 
