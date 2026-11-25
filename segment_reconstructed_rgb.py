"""Run from the suite root: python examples/segment_reconstructed_rgb.py --help.
This uses an EXISTING trained weight; it never fine-tunes at inference time.
"""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from segmentation.api import PlantSegSegmenter
from PIL import Image
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--rgb',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
a=p.parse_args();model=PlantSegSegmenter(a.checkpoint,device=a.device)
# Raw RGB from original/bicubic/frozen ResShift/fine-tuned ResShift. Not an overlay.
result=model.predict(a.rgb)
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
Image.fromarray(result.class_ids).save(out/'raw_class_ids.png')
Image.fromarray(((result.class_ids>0)*255).astype(np.uint8)).save(out/'visible_disease_binary.png')
print('Saved prediction at the supplied RGB dimensions; no ground truth was used.')
