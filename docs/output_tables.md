# Output tables

The command-line workflow writes CSV files and one workbook.

## Core output tables

- `dataset_overview.csv`: sample counts and feature-set dimensions.
- `sample_validation_manifest.csv`: recorded source labels and validation-block labels used for block-aware evaluation.
- `validation_block_source_counts.csv`: recorded source composition of acquisition-sequence validation blocks.
- `source_classification_metrics.csv`: fold/repeat-level metrics for full spectral classifiers.
- `source_classification_summary.csv`: summarized source-classification performance.
- `metadata_negative_control_metrics.csv`: fold/repeat-level metrics for non-spectral metadata controls.
- `metadata_negative_control_summary.csv`: summarized metadata-control performance.
- `minimal_band_metrics.csv`: fold/repeat-level metrics for sparse derivative-band classifiers.
- `minimal_band_summary.csv`: summarized sparse classifier performance.
- `minimal_band_selection_stability.csv`: derivative bands selected across fold-wise feature selection and bootstrap stability selection.
- `derivative_band_mapping.csv`: analysis feature IDs mapped to adjacent wavelengths and midpoint wavelengths.
- `full_data_derivative_source_effect_sizes.csv`: descriptive source effect sizes for all reflectance-derivative features.
- `permutation_null_metrics.csv`: label-permutation null distribution for selected classifiers.
- `block_fold_predictions.csv`: sample-level held-out predictions.
- `analysis_summary.md`: compact text summary of core results.
- `Supplementary_Data_S1_optical_passport.xlsx`: multi-sheet supplementary workbook for the optical-passport validation outputs.

## Class encoding

- Niger = 0
- Senegal = 1

Predicted probabilities in output tables are therefore reported as `prob_senegal`.
