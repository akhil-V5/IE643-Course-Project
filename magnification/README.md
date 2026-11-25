# Magnification component

The current project focus is SegFormer training. This folder provides the separately configured reconstruction path for later comparisons and retains an adapted optional training module from the user's earlier code.

- `configs/bicubic.yaml`: deterministic ×4 bicubic.
- `configs/resshift_pretrained.yaml`: official v1 RealSR×4/15step checkpoint,64pixelLR tiles, overlap16, frozen at inference.
- `configs/resshift_finetune.yaml`: sharedsplit PlantSeg training, HR256/LR64, native diffusionMSE +0.1 lesionweighted RGBL1; lesionweight5, AdamW1e-5, physicalbatch1 accumulation4,10,000optimizersteps. These initial settings are not optimality claims.

Use root `download-sr`, `check-sr`, `reconstruct`, `train-sr`, `evaluate` commands. The upstream source and checkpoint files are downloaded separately with recorded commit/hashes. No xformers dependency is imposed. The source API/shape is checked by the adapter rather than accepting arbitrary ResShift checkpoints. Fine-tuned checkpoints must match the shared dataset manifest for the benchmark.

Whole-image inference is tiled, including very small and odd-sized images; outputgrid exactly4×LR then optional known paired-evaluation padding removed. Each sample gets a stable seed derived from sampleID and configuredseed. Threading/hardware can still affect floating-point reproducibility.

## Training monitoring versus final comparison

Optional SR training validates deterministic256pixel patches from validation sources, including padded short sources; it does NOT call SegFormer. It saves RGBs and visible GT lesion/valid masks, and clearly labels overlays as GT (not predictions). Patch monitoring reports legacy valid-pixel PSNR (exact matches capped120dB), fullpatch SSIM and auxiliary errors. These are training diagnostics only. Padding and patchprotocol differ from the common final whole-image benchmark.

For the final report, ALWAYS use root `evaluate`, which recomputes PSNR/SSIM on full unpadded original RGB, predicts lesions with the fixed SegFormer, and reports the four agreed metrics. Do not copy patch-monitoring values into that table. `best.pth` selection by patch lesion-MAE (falling back to overall MAE when no lesion is present) is a monitoring choice; it does not guarantee best lesion localization.

Native loss covers padded context, whereas auxiliary lesion loss excludes invalidpadding. Random patch selection is not lesion-biased. Validationpatch selection never uses predicted disease to drop examples. SR training resume restores updates/RNG but does not promise identical sample order to an uninterrupted multiworker run; this legacy trainer is not distributed.

Standalone reconstruction without a paired reference cannot calculate meaningful PSNR/SSIM. Its outputs are RGB and magnification panels, not predicted lesion masks. Use a subsequent segmenter for disease predictions.

Fixedpatch training does not require equal-sized source files. See `../docs/IMAGE_SIZES.md`.
