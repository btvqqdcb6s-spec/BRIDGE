# BRIDGE
This repository contains all the code to reproduce the experiments for the submitted paper entitled 'CATE Estimation with Expert Knowledge via Prior Regularization'.

## Data

All TCGA experiments require the preprocessed data file `tcga_data/tcga_X_all.npz`. To generate it, run:

```
python get_data.py
```

This downloads RNA-seq and clinical data for BRCA, LUAD, and KIRC from UCSC Xena and writes the preprocessed array to `tcga_data/`.

## Experiments

The results in Section 4.1 (Table 1) are produced by `BRIDGE_tcga.py`, which runs BRIDGE under four prior quality conditions, and `benchmarks_tcga.py`, which runs the context-aware baselines (CFRNet, T-learner, DR-learner, Causal Forest). Both scripts require the TCGA data file above. The results in Sections 4.2 and 4.3 (Figure 3) are produced by `BRIDGE_simulation_recovery.py`, which runs the synthetic block-correlated simulation and saves the two recovery bar charts to `bridge_plots_sim/`. The results in Section 4.4 (Figure 4) are produced by `BRIDGE_simulation_hard_selection.py`, which runs the BRCA-only feature-subset sweep and saves the PEHE-vs-k figure to `bridge_plots_sim/`.
