# CMRV Information-State Audit

Minimal public assets accompanying the manuscript **“When Is ECG Still Worth Fusing? An Information-State Audit of Residual Modality Value and Fusion Failure Across Perioperative Tasks.”**

The repository exposes the formula-level clinical-information-conditioned modality residual value (CMRV) audit and the publication-level aggregates needed to create publication-equivalent versions of the five main figures. It does not expose patient-level data or the governed clinical model-development pipeline; exact journal-layout bytes are not claimed.

## Repository contents

- `src/cmrv_metrics.py` — formula-level implementations of implemented fusion value, oracle potential information value, fusion implementation gap, entropy normalization, and information saturation.
- `scripts/reproduce_main_figures.py` — creates publication-equivalent versions of Figures 1–5 from the public aggregate CSV.
- `data/cmrv_published_aggregate_results.csv` — aggregate estimates and pointwise confidence intervals already reported in the article.
- `requirements.txt` — versions used to verify the public figure-reproduction script.
- `LICENSE` — MIT License for code.
- `LICENSE-DATA` — CC BY 4.0 notice for the aggregate CSV.

## Reproduce the figures

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce_main_figures.py
```

The script writes five PDF and five 300-dpi PNG files to `outputs/`. It uses only the included aggregate CSV and creates no model predictions or patient-level outputs.

The public rendering environment was verified with Python 3.12, NumPy 2.2.5, pandas 2.2.3, and Matplotlib 3.10.3. This is distinct from the frozen analysis environments reported in the manuscript; it must not be interpreted as a replacement execution receipt for model fitting or ECG representation extraction.

## Data availability

The included CSV contains only publication-level aggregate estimates, pointwise confidence intervals, descriptive configuration labels, and frozen gate outcomes already displayed in the manuscript. It contains no patient identifiers, patient-level data, embeddings, predictions, fitted coefficients, or model checkpoints.

The source clinical datasets are governed separately:

- VitalDB is publicly accessible subject to its terms of use.
- MIMIC-IV and MIMIC-IV-ECG are available through PhysioNet credentialing, required training, and the applicable data-use agreement.
- Patient-level source data and derived patient-level artifacts cannot be redistributed through this repository.

## Scope and caveats

- CMRV is a task-level information-state audit, not a deployable clinical model.
- The six task-stage intervals and 24-cell configuration map use pointwise inference; they are not simultaneous family-wise claims.
- Clean cross-task transport passed, but the prespecified engineering stress gate failed.
- Physiological-domain localization remained partial.
- The reported lower-tier policy result is descriptive and is not a universal recommendation to acquire, omit, or act on ECG data.
- New datasets, representations, or deployment settings require local validation and, where relevant, recalibration.

## Licenses

Code is released under the MIT License. `data/cmrv_published_aggregate_results.csv` is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0). The licenses do not alter the access terms of VitalDB, MIMIC-IV, MIMIC-IV-ECG, or any other source dataset.

## Citation

Please cite the accompanying manuscript. Author, journal, year, DOI, and final bibliographic details will be added after publication; no DOI has been assigned at the time of this public release.
