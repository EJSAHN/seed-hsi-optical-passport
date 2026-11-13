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

Processed inputs, analysis settings, and the script for reproducing the reported core tables are provided in **Supplementary Data S2 (Reproducibility inputs)**, `Supplementary_Data_S2_Reproducibility_Inputs.zip`, accompanying the study. These inputs are distributed separately from this code repository. Use the supplied `processed_inputs` directory unchanged when following the reported-table instructions below.

For other runs, place the processed source tables in a single input directory. The command-line tool searches both the input directory itself and common subdirectories.

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

## Reproduce the reported core tables

The analysis code used for the reported tables is recorded at commit `b31dd9f81c65cd2e57e8cc69f615cb475c7fedd6`. Install that version of this package, then extract Supplementary Data S2. In the same Python environment, run the following command from the extracted directory containing `run_reported_tables.py`:

```text
python run_reported_tables.py --output results
```

The script reads the supplied inputs and settings and fixes the metadata-control predictor order to the order used for the reported estimates. It does not change predictor values, source labels, validation splits, model settings, or spectral analyses. Use this script rather than the generic command below when reproducing the reported metadata-control metrics; changing predictor order can produce small numerical differences. Supplementary Data S2 documents the ordering and numerical comparison.

An installation and input check is available with `--fast`; it does not use the full analysis settings.

## Run the analysis with other processed inputs

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

All random procedures use an explicit seed. The reported core analysis uses 50 repeated random stratified five-fold cross-validations, 200 label permutations with two repeated CV runs per permutation, 300 class-stratified subsampling iterations without replacement, and minimal-band models with 1, 3, 5, 10, and 20 selected derivative bands.

The command-line option retains the name `--bootstrap` for compatibility, but it specifies **subsampling without replacement**, not bootstrap resampling with replacement. Each iteration samples approximately 80% of each source class and records the ten highest-ranked derivative features. Reported selection counts combine these selections with selections across all minimal-band training folds and model sizes. They are therefore aggregate counts, not counts out of 300 or selection probabilities.

For the reported estimates, use the settings and metadata-column ordering supplied with Supplementary Data S2 as described above.

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
