# Input tables

The analysis starts from processed tables rather than raw acquisition files.

## Required tables

### `hsi_sample_manifest.csv`

Required columns:

- `sample_id`: accession identifier.
- `source`: recorded accession-source label. Values beginning with `N` are encoded as Niger and values beginning with `S` are encoded as Senegal.
- `acquisition_order`: numeric acquisition-sequence order used to construct five-accession validation blocks.

For compatibility with earlier processed exports, the workflow also accepts `country` as an alias for `source` and `hsi_index` as an alias for `acquisition_order`.

Additional columns are ignored unless used by downstream metadata controls.

### `master_hsi_wide_F.csv`

A wide fluorescence matrix. The first required column is `sample_id`; remaining spectral columns should be numeric wavelength labels or column names containing wavelength values.

### `master_hsi_wide_R.csv`

A wide reflectance matrix with the same structure as the fluorescence matrix.

## Optional metadata-control tables

### `pixel_file_parse_report.csv`

If present, numeric file-level summaries containing tokens such as `row`, `pixel`, `size`, `bytes`, or `n_` are aggregated by sample and used as non-spectral metadata negative controls.

### `pixel_feature_coverage.csv`

If present, numeric pixel-feature coverage summaries are included in the metadata-control feature set.

## Notes

Raw hyperspectral image files, raw pixel-level spreadsheets, and raw variant files are not required for this workflow. They are not needed to reproduce the reported validation tables once the processed matrices and manifest files are provided.
