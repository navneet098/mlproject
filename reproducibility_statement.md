# Reproducibility statement

WattWise is designed to reproduce the forecasting and anomaly-detection experiments from source. The repository records dependency versions, a fixed random seed, data provenance, preprocessing rules, chronological split proportions, model hyperparameters, and command-line execution steps.

A reproduction requires Python 3.10/3.11 and access to the official UCI dataset URL (or a manually downloaded copy at `data/raw/household_power_consumption.txt`). Run `pip install -r requirements.txt` followed by `python -m src.run_pipeline`. The pipeline writes model files, test metrics, prediction/anomaly CSV files, and figures. The Streamlit dashboard reads those outputs.

The final 15% of the time-ordered supervised examples is held out for test reporting. Models are fitted on the first 70%; the validation segment is used to fit the forecast-residual Isolation Forest. The test segment is not used to fit forecasting models or anomaly detectors.

**Status:** This statement describes the reproducibility design and execution procedure. A completed empirical reproducibility claim should only be made after the pipeline has been run successfully from a clean environment and the resulting metrics and artifact hashes have been archived. The repository starter package does not claim that this execution has already occurred.

Known constraints include possible changes in upstream hosting, platform/library differences, and the single-household nature of the dataset. Exact bitwise identity across all platforms is not guaranteed; numerical agreement should be assessed using the generated metrics and recorded environment.
