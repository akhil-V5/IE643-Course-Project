# PlantSeg Experiment Suite v3

**Fine-tune SegFormer on your existing splits; compare original, bicubic, pretrained ResShift and fine-tuned ResShift using the same held-out cases.**

This is an updated research implementation for the supplied course project, not a trained model or a claimed reproduction of a published score. It adapts the user's previously supplied segmentation and reconstruction code. See `docs/TEST_REPORT.md` for what was actually tested.

## 1. The experiment supported

1. Audit your existing train, validation and test memberships, without creating a new split.
2. Initialize SegFormer from NVIDIA's ADE20K checkpoint, replace its classification head and fine-tune on original PlantSeg training images/masks. Select a checkpoint using validation only.
3. Freeze that one segmenter for every reconstruction condition:
   - **original**: original RGB → SegFormer;
   - **bicubic**: common synthetic LR → bicubic ×4 → SegFormer;
   - **resshift-pretrained**: that same LR → frozen official ResShift → SegFormer;
   - **resshift-finetuned**: that same LR → later PlantSeg-fine-tuned ResShift → SegFormer.
4. Predict a NEW disease mask from each RGB image. No disease gate and no enlarged old disease mask.

The four headline metrics are **PSNR, SSIM, foreground multiclass mIoU and binary disease Dice/F1**. Ground truth is not an inference input.

`evaluate` is a benchmark with paired references. `infer-seg` and `infer` are deployment-style commands without an assumed reference; they cannot invent ground-truth metrics.

## 2. Structure

```text
run.py                           One logged command entry point
configs/data.yaml                Your exact existing splits
configs/evaluation.yaml          Shared test/validation protocol
common/                          Split audit, hashes and shared utilities
segmentation/
  configs/segformer_b0.yaml       Lower-compute option
  configs/segformer_b1.yaml       Intermediate option
  configs/segformer_b2.yaml       Main starting configuration
  train_segmentation.py          Fine-tuning and resume
  infer.py                      File/folder/RGB inference
  api.py                        Python interface for later frame modules
  plantseg_seg/                  Model, data, metrics, training, visualizations
magnification/
  configs/bicubic.yaml           Simple interpolation
  configs/resshift_pretrained.yaml   Frozen reconstruction configuration
  configs/resshift_finetune.yaml     Optional later reconstruction training
  reconstruct.py                Native-size tiled reconstruction + provenance
  resshift/                     User's adapted ResShift wrapper/trainer
experiments/                     Shared LR preparation and fixed-segmenter scoring
tests/                           CPU tests and optional real-Transformers API check
docs/                            Protocol, sizes, tests, changes and source notes
```

Run commands from this folder. The router also sets the working directory to this suite root, so relative paths in configurations mean this root, NOT the directory containing a YAML file. Use absolute paths for your dataset and Python executables.

## 3. Environments

Recommended: **Python 3.11**, separate SegFormer and ResShift environments. Do not install both requirement files into one existing research environment blindly. PyTorch/torchvision must be a compatible pair for your CUDA driver. Install them using the official PyTorch installation instructions before the requirements below.

Example environment creation (Linux/macOS shell; `conda` is optional):

```bash
conda create -n plantseg-seg python=3.11 -y
conda activate plantseg-seg
# Install a compatible torch>=2.6 and torchvision build for your GPU here.
python -m pip install -r requirements-segmentation.txt
python run.py check-seg --device cpu
```

The segmentation implementation targets Transformers 4.46.3. That is an intentional API pin, not a statement that it is the latest version. This package's CPU checks were run in a different, preinstalled environment; the exact GPU dependency combinations have NOT been executed here.

Only when starting the ResShift experiments:

```bash
conda create -n plantseg-sr python=3.11 -y
conda activate plantseg-sr
# Install a compatible torch>=2.6 and torchvision build for your GPU here.
python -m pip install -r requirements-magnification.txt
```

You can keep the segmentation environment active and pass the other interpreter:

```bash
python run.py --sr-python /absolute/path/to/plantseg-sr/bin/python download-sr
```

Global `--seg-python` and `--sr-python` go **before** the subcommand. The router runs the two model environments in separate processes. Without these options it uses the current interpreter for both.

For a controller-only environment, install `requirements-control.txt` and supply both interpreters. If running native Windows, replace the executable path with the relevant environment's `python.exe`.

## 4. Use your existing dataset splits

Edit ONLY the six image/mask paths in `configs/data.yaml` initially. All three splits must be present and nonempty. Example:

```yaml
prepared: data/prepared
preview_count: 12
allow_binary_labels: false
source:
  mode: folders
  splits:
    train:
      images: /data/PlantSeg/images/train
      masks: /data/PlantSeg/annotations/train
    val:
      images: /data/PlantSeg/images/val
      masks: /data/PlantSeg/annotations/val
    test:
      images: /data/PlantSeg/images/test
      masks: /data/PlantSeg/annotations/test
```

Each image and mask must have the same relative stem. For different layouts, use:
- `configs/data_csv.example.yaml`: existing CSV columns `image,mask,split`, optional unique `id`. Relative paths resolve from the CSV directory. Split values are exactly `train,val,test`.
- `configs/data_lists.example.yaml`: existing train/val/test lists. Each line identifies an exact relative image filename or stem; no guessing of adjacent or similarly named files.

For a custom data config, pass `--data-config path/to/data.yaml` consistently to preparation, training and evaluation.

```bash
python run.py prepare --data-config configs/data.yaml
python run.py audit --data-config configs/data.yaml --image-size 512
```

### Annotation contract

Use original indexed masks: **0=background/non-disease, 1..115=original disease category IDs, 255=ignore**. See `segmentation/labels/plantseg_classes.json`. The model has 116 output channels. Do not substitute binary masks or coloured display masks. No ADE20K label reduction is applied: zero stays zero.

Palette PNGs are read as their stored indices, not converted to grayscale colours. Arbitrary RGB label maps require an explicit verified conversion outside this package. Dataset name alone does not prove category-map compatibility.

Preparation checks every pair, category range, image/mask geometry, exact duplicates and decoded content. It **never** resplits, rescales or silently deletes cases. Cross-split duplicate images, conflicting annotations or unusable inputs generate an error/report for explicit resolution. Same-split identical pairs are retained and reported. Exact hashing does not detect all near-duplicate photographs or related plants; assess those separately.

No minimum source image dimension is imposed. Source files remain unchanged. EXIF orientation is not independently transformed: your RGB and annotation must already share stored pixel coordinates. Resolution audit reports symptoms/classes disappearing when resized to the segmenter canvas.

## 5. Download initialization weights and train SegFormer

```bash
python run.py download-seg --config segmentation/configs/segformer_b2.yaml
python run.py check-seg --config segmentation/configs/segformer_b2.yaml --check-model --device cuda
python run.py train-seg --config segmentation/configs/segformer_b2.yaml --run runs/segformer_b2
```

`check-model` loads the real downloaded architecture and does a 128-pixel forward/backward mechanics check. It is not an accuracy test. The initializer is ADE20K-trained, NOT already PlantSeg-trained. Only the final classifier is replaced; the default then fine-tunes the full encoder and decoder.

### Deliberate initial recipe

| Setting | Default and rationale |
|---|---|
| Canvas | 512×512, aspect-preserving resize/padding, matching your earlier setup |
| Optimizer | AdamW; encoder LR 6e-5, decoder LR 6e-4 |
| Loss | Cross-entropy, ignore255; optional Dice loss disabled |
| Batch | 2, accumulation1; adjust according to GPU memory |
| Training limit | 30 maximum epochs, early stopping after5 non-improving validation events |
| Validation | Entire supplied validation split at each epoch by default |
| Checkpoint selection | Foreground multiclass mIoU |
| Augmentation | Paired horizontal/vertical flips; no synthetic degradation by default |
| Precision | CUDA automatic mixed precision; can disable with `--no-amp` |

These are documented starting settings, not a tuned or published-optimal PlantSeg recipe. Use validation for changes; do not tune on test.

Lower-memory run, same effective batch:

```bash
python run.py train-seg --config segmentation/configs/segformer_b2.yaml \
  --run runs/segformer_b2_accum --batch-size 1 --accumulate 2
```

Short mechanics run (NOT a final benchmark):

```bash
python run.py train-seg --config segmentation/configs/segformer_b0.yaml \
  --run runs/debug_b0 --steps 20 --workers 0 --val-limit 8
```

Download the B0 initializer first when changing architecture. B1 has its own config as well. CLI overrides include `--epochs`, `--steps`, `--image-size`, `--workers`, `--preview-count`, `--val-limit`, `--trainable all|decoder`, `--device`, and `--no-amp`. Set `--val-limit 0` for full validation and `--preview-count 0` for all panels. Detailed individual masks are saved for all evaluated images by default even when panel count is12.

Resume with the same split and core settings:

```bash
python run.py train-seg --run runs/segformer_b2 --resume runs/segformer_b2/last.pth
```

Epochs/steps specify the TOTAL budget, not additional work. `--init` starts a new optimizer/run from a PlantSeg checkpoint; `--resume` continues optimizer/scaler/RNG/sampler state. Extending a schedule changes the experiment. GPU kernels need not be bitwise reproducible across different systems.

## 6. Validate or test the fixed segmenter

```bash
python run.py evaluate --checkpoint runs/segformer_b2/best.pth \
  --split val --modes original --output results/b2_val_original

python run.py evaluate --checkpoint runs/segformer_b2/best.pth \
  --split test --modes original --output results/b2_test_original
```

Use `--split val` for development. Once selected, keep the SAME checkpoint and inference canvas for every test condition. The evaluator verifies actual source/annotation hashes and the checkpoint's training manifest.

## 7. Bicubic versus the original

```bash
python run.py evaluate --checkpoint runs/segformer_b2/best.pth \
  --split test --modes original bicubic --output results/original_bicubic
```

This creates the common low-resolution cohort once. `limit: 0` in `configs/evaluation.yaml` means ALL declared split members. Images are not filtered using their predicted disease status.

### Odd sizes and small images: no silent deletion

For a **901×603** source, the common preparation replicates only the right/bottom RGB edge to **904×604**, downsamples to **226×151**, and reconstructs to **904×604**. The added padding is removed before segmentation and scoring, restoring **901×603**. The original lesion mask stays **901×603 throughout**.

This retains original border pixels and all source members. It does not create a high-resolution target by enlarging a small source. SSIM is mathematically unsupported for images shorter than3 pixels; these receive an explicit undefined SSIM but remain in all other applicable evaluations. Counts are reported.

## 8. Frozen/pretrained ResShift plus the same segmenter

After installing the separate reconstruction environment:

```bash
python run.py --sr-python /path/to/sr/bin/python download-sr
python run.py --sr-python /path/to/sr/bin/python check-sr

python run.py --sr-python /path/to/sr/bin/python evaluate \
  --checkpoint runs/segformer_b2/best.pth --split test \
  --modes original bicubic resshift-pretrained \
  --output results/initial_three_conditions
```

The bootstrap obtains official source and v1 RealSR ×4 /15-step weights plus the VQ autoencoder. It records the upstream commit and file hashes. Respect upstream licenses. Downloads are external dependencies, not included in this archive.

`check-sr` is a real native sampling and gradient mechanics check using the shared prepared data; it needs CUDA. It does not require a fine-tuned SR checkpoint. Run this BEFORE committing to an SR training run.

The reference ResShift configuration uses **256×256 HR /64×64 LR patches**. That is a training/sample shape constraint, not a restriction on whole source image dimensions. The wrapper pads short LR inputs, reconstructs overlapping64-pixel tiles, blends them and returns the exact ×4 native size. It does not shrink a complete LR image to64×64. The backend, VQ and source signatures must match.

The pretrained RealSR model was not trained specifically for your pure-bicubic PlantSeg experiment. Treat its result as the off-the-shelf baseline it is, without assuming it will beat bicubic.

## 9. Later: optional PlantSeg ResShift fine-tuning

The focus now is SegFormer. The future SR trainer is kept in `magnification/resshift` with a separate config and the SAME shared source split:

```bash
python run.py --sr-python /path/to/sr/bin/python train-sr \
  --config magnification/configs/resshift_finetune.yaml \
  --mode disease-aware --run runs/resshift_plantseg
```

Larger sources provide aligned256-pixel patches. Smaller sources are edge-padded; annotation padding is invalid for the auxiliary loss. No source is excluded purely for size. Native diffusion loss still includes padded context. Monitoring is deterministic PATCH evaluation, not the final whole-image benchmark; see `magnification/README.md`.

Then:

```bash
python run.py --sr-python /path/to/sr/bin/python evaluate \
  --checkpoint runs/segformer_b2/best.pth --split test --modes all \
  --sr-checkpoint runs/resshift_plantseg/best.pth \
  --output results/final_four_conditions
```

Both SR checkpoints and the segmenter are frozen during evaluation. `--modes all` fails without a verified fine-tuned SR checkpoint; it never substitutes pretrained weights silently. Use explicit first-three modes until fine-tuning is complete.

An optional baseline SR fine-tuning condition (`--mode baseline`) disables the lesion-weighted RGB auxiliary loss. Do not call pretrained-vs-fine-tuned improvement proof of lesion weighting alone: that comparison also changes domain adaptation.

## 10. Standalone inference and use with previous modules

### Segment already-produced RGB: DO NOT magnify again

```bash
python run.py infer-seg --checkpoint runs/segformer_b2/best.pth \
  --input /path/to/raw_reconstruction_rgb_only --output predictions/existing_rgb
```

Works for one file or recursively collected images. Use an RGB-only directory, not a directory mixing RGB, masks, panels and overlays. Output cannot be nested inside the input directory. Arbitrary source dimensions are accepted and predictions are restored to them.

### Actual LR input → selected magnifier → SegFormer

```bash
python run.py infer --mode bicubic --checkpoint runs/segformer_b2/best.pth \
  --input /path/to/low_resolution_leaf.png --output predictions/bicubic_leaf

python run.py --sr-python /path/to/sr/bin/python infer \
  --mode resshift-pretrained --checkpoint runs/segformer_b2/best.pth \
  --input /path/to/low_resolution_leaf.png --output predictions/resshift_leaf
```

`infer` does NOT perform another synthetic downsampling: its source is already LR. For fine-tuned SR use `--mode resshift-finetuned --sr-checkpoint runs/resshift_plantseg/best.pth`.

### Magnification only

```bash
python run.py reconstruct --mode bicubic --input /path/to/lr_rgb --output magnified/bicubic
python run.py --sr-python /path/to/sr/bin/python reconstruct \
  --mode resshift-pretrained --input /path/to/lr_rgb --output magnified/resshift
```

### Python integration (load once)

```python
from segmentation.api import PlantSegSegmenter
segmenter = PlantSegSegmenter('runs/segformer_b2/best.pth', device='cuda')
result = segmenter.predict(reconstructed_rgb)  # PIL, RGB uint8 array, or path
mask = result.class_ids                      # input H×W, IDs0..115
```

OpenCV BGR must be converted to RGB first. See `examples/segment_reconstructed_rgb.py`. An optional leaf silhouette must already match the reconstructed image grid; it is NOT a disease annotation.

Your old final-frame driver predicts before reconstruction. This suite does not modify that old directory: its disease stage must be connected after reconstruction when you resume frame/video integration.

Without a GT mask, deployment inference has no IoU/Dice. Without paired original HR, it has no reference PSNR/SSIM. Use `evaluate` for comparable benchmark metrics. An optional `infer-seg --mask-dir` supports explicitly aligned local masks but does not turn arbitrary inputs into the locked test cohort.

## 11. Results, logs and readable masks

Every routed process saves a console log and command/exit metadata in `logs/`. Output folders also receive a process log. Use a new output directory for each evaluation; the immutable LR cache is reused and verified.

Training creates `config.yaml`, `training.jsonl`, `validation.jsonl`, CSV summaries, `best.pth`, `last.pth` and `validation/step_XXXXXX/`. JSONL records are written throughout a run; consolidated CSV is written on normal completion. If interrupted, JSONL and periodic checkpoints remain useful.

Each evaluated case saves:
- `class_ids/` and `gt_class_ids/`: unchanged raw category values for computation; they can look dark in a viewer and are NOT display masks.
- `disease_binary/` and `gt_disease_binary/`: **white disease, black background**, ignored GT mid-gray127.
- `gt_valid/`: white valid annotation, black ignored annotation.
- `class_colors/`, `gt_class_colors/`, `overlays/`, `gt_overlays/`: class distinctions and RGB context.
- `probability/`: 16-bit disease probability display data, NOT severity.
- `panels/` or `preview/`: labelled RGB + GT + predicted comparisons. Previews can be resized; titles state native sizes. Masks use nearest-neighbour display resizing and visible borders.

Validation saves all individual masks by default. `preview_count:12` limits only large panels; `0` saves all panels. Comparison panels show the same representative cases for all conditions. This can consume substantial storage when all validation epochs and conditions are retained.

Evaluation writes `summary.csv/json` at the requested result root, detailed metrics and per-image/per-class CSVs under `evaluation/`, confusion matrices and inference provenance. Reconstruction folders include LR/RGB and paired-reference panels. Pure reconstruction never pretends a GT overlay is a predicted lesion mask.

## 12. Scoring rules and reproducibility

- RGB PSNR/SSIM: compare each method's unpadded RGB to the original image, values[0,1], no border shave; average image-level scores. Original self-comparison is left blank.
- Foreground mIoU: pooled original-grid confusion over valid pixels; average disease classes with nonzero GT-or-prediction union. Background excluded only from class average, so false positives on background still count.
- Binary disease Dice: all classes1..115 merged; pooled valid-pixel TP/FP/FN. Empty pooled foreground is undefined, not silently perfect.
- Scores are fractions0..1, not percentages. Detailed policy and evaluated-image counts are saved.
- Same source manifest, same ordered evaluation IDs, exact same LR bytes, original masks, segmenter checkpoint, canvas and normalization across conditions.
- Missing/changed source or reconstruction outputs, mismatched dimensions, mismatched fine-tuning split signatures and incorrectly labelled pretrained/fine-tuned conditions fail explicitly.

You may rerun only one condition (`--modes resshift-pretrained`) or several. For reuse of an existing suite reconstruction, `--variant resshift-pretrained=/absolute/path/to/manifest.json` avoids rerunning SR after verifying provenance. This is the v3 reconstruction manifest, not an arbitrary old JSON or filename guess.

## 13. Limits and testing

No PlantSeg images, trained checkpoint, drone UI, severity model or upstream ResShift source/weights are bundled. CPU tests use synthetic images and small stand-in networks where dependencies are absent. Passing them does not establish GPU execution, learning convergence or accuracy. The actual SegFormer/ResShift checks on your machine remain essential.

See `docs/TEST_REPORT.md`, `docs/IMAGE_SIZES.md`, `docs/CHANGELOG.md` and `docs/SOURCES.md`. All original uploads remain unchanged.
