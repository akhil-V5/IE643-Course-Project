# IE643 Course Project 
# Disease-Aware Diffusion for Fine-Grained Crop Disease Localization from Drone-Captured Video

**Fine-tune SegFormer on your existing splits; compare original, bicubic, pretrained ResShift and fine-tuned ResShift using the same held-out cases.**

## 1. Experiments

1. Audit your existing train, validation and test memberships, without creating a new split.
2. Initialize SegFormer from NVIDIA's ADE20K checkpoint, replace its classification head and fine-tune on original PlantSeg training images/masks. Select a checkpoint using validation only.
3. Freeze that one segmenter for every reconstruction condition:
   - **original**: original RGB → SegFormer;
   - **bicubic**: common synthetic LR → bicubic ×4 → SegFormer;
   - **resshift-pretrained**: that same LR → frozen official ResShift → SegFormer;
   - **resshift-finetuned**: that same LR → later PlantSeg-fine-tuned ResShift → SegFormer.
4. Predict a NEW disease mask from each RGB image.

The four headline metrics are **PSNR, SSIM, foreground multiclass mIoU and binary disease mIoU. 


## 2. Environments

Recommended: **Python 3.11**, 

Segformer setup
```bash
conda create -n plantseg-seg python=3.11 -y
conda activate plantseg-seg
# Install a compatible torch>=2.6 and torchvision build for your GPU here.
python -m pip install -r requirements-segmentation.txt
python run.py check-seg --device cpu
```
ResShift setup:

```bash
conda create -n plantseg-sr python=3.11 -y
conda activate plantseg-sr
# Install a compatible torch>=2.6 and torchvision build for your GPU here.
python -m pip install -r requirements-magnification.txt
```

## 3. Use your existing dataset splits

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


## 4. Download initialization weights and train SegFormer

```bash
python run.py download-seg --config segmentation/configs/segformer_b2.yaml
python run.py train-seg --config segmentation/configs/segformer_b2.yaml --run runs/segformer_b2
```

The initializer is ADE20K-trained, NOT already PlantSeg-trained. Only the final classifier is replaced; the default then fine-tunes the full encoder and decoder.



## 5. Validate or test the fixed segmenter

```bash
python run.py evaluate --checkpoint runs/segformer_b2/best.pth \
  --split val --modes original --output results/b2_val_original

python run.py evaluate --checkpoint runs/segformer_b2/best.pth \
  --split test --modes original --output results/b2_test_original
```



## 6. Bicubic versus the original

```bash
python run.py evaluate --checkpoint runs/segformer_b2/best.pth \
  --split test --modes original bicubic --output results/original_bicubic
```

## 7. Frozen/pretrained ResShift plus the same segmenter

After installing the separate reconstruction environment:

```bash
python run.py --sr-python /path/to/sr/bin/python download-sr
python run.py --sr-python /path/to/sr/bin/python check-sr

python run.py --sr-python /path/to/sr/bin/python evaluate \
  --checkpoint runs/segformer_b2/best.pth --split test \
  --modes original bicubic resshift-pretrained \
  --output results/initial_three_conditions
```

## 8. PlantSeg ResShift fine-tuning



```bash
python run.py --sr-python /path/to/sr/bin/python train-sr \
  --config magnification/configs/resshift_finetune.yaml \
  --mode disease-aware --run runs/resshift_plantseg
```


Evaluation:

```bash
python run.py --sr-python /path/to/sr/bin/python evaluate \
  --checkpoint runs/segformer_b2/best.pth --split test --modes all \
  --sr-checkpoint runs/resshift_plantseg/best.pth \
  --output results/final_four_conditions
```




## 9. Scoring rules and reproducibility

- RGB PSNR/SSIM: compare each method's unpadded RGB to the original image
- Foreground mIoU: pooled original-grid confusion over valid pixels; average disease classes with nonzero GT-or-prediction union. 
- Binary disease Dice: all classes1..115 merged; pooled valid-pixel TP/FP/FN. Empty pooled foreground is undefined, not silently perfect.






