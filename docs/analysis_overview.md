# Analysis overview

This repository reproduces the core optical-passport validation workflow from processed accession-level seed HSI tables.

The workflow performs four main steps:

1. Align fluorescence and reflectance HSI matrices to the accession manifest.
2. Construct reflectance first-derivative features and map each feature to adjacent wavelengths and midpoint wavelength.
3. Evaluate accession-source classification under random stratified validation, acquisition-block-holdout validation, and source-specific block-holdout validation.
4. Summarize sparse derivative-band performance, metadata-only negative controls, label-permutation null distributions, and held-out source probabilities.

The workflow does not perform new SNP variant calling, genome-wide association analysis, or disease-trait modeling. Genotype-aware trait-audit and genomic-audit results are provided as processed tables in Supplementary Data S1.
