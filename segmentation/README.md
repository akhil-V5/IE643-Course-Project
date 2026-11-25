# Segmentation component

Use the root `run.py`, not old v2 script paths. Segmentation and reconstruction dependencies are independent. Fine-tune from ADE20K initialization using the shared audited source manifest; select using validation; freeze the final checkpoint for all experiment branches.

B2 default:512canvas, CE, allparameters, AdamW encoder6e-5/decoder6e-4, batch2, max30epochs/earlystop5, fullvalidation. B0/B1 alternative configurations. All settings and overrides are saved. No disease gate in the main workflow.

`train-seg`, `infer-seg`, `evaluate`, `audit`, `download-seg`, `check-seg` are the relevant commands. See root README/QUICKSTART. `api.py` offers `PlantSegSegmenter` accepting PIL/RGBuint8/path, returning native-grid masks. Optional original leaf mask may restrict output but is never a lesion annotation.

Raw IDs preserve semantic categories. `disease_binary` and `gt_disease_binary` use white/black; ignored GT is127 and `gt_valid` disambiguates it. Display scaling never rewrites the mask used in metrics.

Validation saves every evaluated mask by default and representative panels, with all metrics over the full split unless an explicit limit is used. The independent evaluator keeps the original ground-truth grid unchanged and compares every RGB condition using identical preprocessing and one loaded model.
