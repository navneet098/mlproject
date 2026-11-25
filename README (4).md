# Checkpoints

The pipeline writes fitted model artifacts to `models/`. For submission, first run the pipeline and inspect file sizes.

- If artifacts are small enough for the repository, intentionally add the selected `.joblib` files (overriding the ignore rule).
- If too large, upload them to a model hosting service such as Hugging Face Hub. Add the exact repository URL, immutable revision/commit, filenames, SHA-256 hashes, and download instructions here.
- Do not leave a fabricated model URL. Replace this note only after a real upload.

No trained checkpoint is claimed to be included in this source-only starter package.
