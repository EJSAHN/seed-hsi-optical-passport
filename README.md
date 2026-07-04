# Seed HSI Optical Passport for Sorghum Germplasm

This repository provides the analysis workflow used to generate tabular validation outputs for a seed hyperspectral imaging (HSI) study of Nigerien and Senegalese sorghum germplasm. The workflow tests whether seed fluorescence and reflectance spectra support an optical passport for accession source under leakage-safe and block-aware validation.

## Scientific purpose

The workflow evaluates four questions:

1. Do seed HSI spectra classify recorded accession-source labels under leakage-safe validation?
2. Does source classification remain strong when acquisition-sequence validation blocks are held out?
3. Can a sparse reflectance-derivative barcode reproduce the source-classification signal?
4. Is the spectral classifier stronger than metadata-only negative controls?

This repository reproduces the core optical-passport source-classification, sparse-barcode, metadata-control, and permutation tables from processed seed HSI input tables. Genotype-aware trait-audit and genomic-audit source tables are provided as processed outputs in Supplementary Data S1.

## Required input files

Place the processed source tables in a single input directory. The command-line tool searches both the input directory itself and common subdirectories.

Required:

- `hsi_sample_manifest.csv`
- `master_hsi_wide_F.csv`
- `master_hsi_wide_R.csv`

Optional but recommended:

- `pixel_file_parse_report.csv`
- `pixel_feature_coverage.csv`

The raw pixel-level HSI exports are not required for this reproducibility workflow. The workflow starts from processed accession-level HSI matrices and sample manifests.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Conda users can also create the environment from:

```bash
conda env create -f environment.yml
conda activate seed-hsi-passport
python -m pip install -e .
```

## Run the analysis

```bash
seed-hsi-passport \
  --input data/processed \
  --output results/optical_passport \
  --repeats 50 \
  --permutations 200 \
  --permutation-repeats 2 \
  --bootstrap 300 \
  --band-counts 1,3,5,10,20
```

The same command can be run as a module:

```bash
python -m seed_hsi_optical_passport \
  --input data/processed \
  --output results/optical_passport
```

## Main outputs

- `dataset_overview.csv`
- `sample_validation_manifest.csv`
- `validation_block_source_counts.csv`
- `source_classification_metrics.csv`
- `source_classification_summary.csv`
- `metadata_negative_control_metrics.csv`
- `metadata_negative_control_summary.csv`
- `minimal_band_metrics.csv`
- `minimal_band_summary.csv`
- `minimal_band_selection_stability.csv`
- `derivative_band_mapping.csv`
- `full_data_derivative_source_effect_sizes.csv`
- `permutation_null_metrics.csv`
- `block_fold_predictions.csv`
- `analysis_summary.md`
- `Supplementary_Data_S1_optical_passport.xlsx`

## Validation design

The full spectral classifiers use L2-regularized logistic regression with class-balanced weights. Missing values are median-imputed and quantitative predictors are standardized within the training fold. The block-aware classifiers use the same preprocessing and model structure, but test sets are defined by acquisition-sequence validation blocks instead of random sample partitions.

Reflectance-derivative features are computed from adjacent reflectance bands. Feature names preserve an analysis feature ID and the midpoint wavelength. Article figures can report the midpoint wavelength or adjacent-band interval, while Supplementary Data S1 preserves feature IDs for computational traceability.

## Reproducibility notes

All random procedures use an explicit seed. The default settings match the reported validation analysis: 50 repeated random stratified cross-validations, 200 label permutations with two repeated CV runs per permutation, 300 bootstrap stability iterations, and minimal-band models with 1, 3, 5, 10, and 20 selected derivative bands.

## Repository structure

```text
seed-hsi-optical-passport/
  README.md
  pyproject.toml
  requirements.txt
  environment.yml
  data/README.md
  docs/
    analysis_overview.md
    input_tables.md
    output_tables.md
  src/seed_hsi_optical_passport/
    cli.py
    features.py
    io.py
    metrics.py
    validation.py
    workbook.py
  tests/
```

## License

Code is provided under the MIT License unless the final data-release repository uses a different institutional license.
