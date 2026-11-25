"""Class order from tqwei05/PlantSeg/mmseg/datasets/plantseg115.py.

Index 0 is named 'background' here (the original class name is an empty string).
This changes only the display name, NOT the numerical label.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np

CLASS_NAMES = (
    'background', 'apple black rot', 'apple mosaic virus', 'apple rust', 'apple scab',
    'banana anthracnose', 'banana black leaf streak', 'banana bunchy top',
    'banana cigar end rot', 'banana cordana leaf spot', 'banana panama disease',
    'basil downy mildew', 'bean halo blight', 'bean mosaic virus', 'bean rust',
    'bell pepper bacterial spot', 'bell pepper blossom end rot',
    'bell pepper frogeye leaf spot', 'bell pepper powdery mildew',
    'blueberry anthracnose', 'blueberry botrytis blight', 'blueberry mummy berry',
    'blueberry rust', 'blueberry scorch', 'broccoli alternaria leaf spot',
    'broccoli downy mildew', 'broccoli ring spot', 'cabbage alternaria leaf spot',
    'cabbage black rot', 'cabbage downy mildew', 'carrot alternaria leaf blight',
    'carrot cavity spot', 'carrot cercospora leaf blight',
    'cauliflower alternaria leaf spot', 'cauliflower bacterial soft rot',
    'celery anthracnose', 'celery early blight', 'cherry leaf spot',
    'cherry powdery mildew', 'citrus canker', 'citrus greening disease',
    'coffee berry blotch', 'coffee black rot', 'coffee brown eye spot',
    'coffee leaf rust', 'corn gray leaf spot', 'corn northern leaf blight',
    'corn rust', 'corn smut', 'cucumber angular leaf spot', 'cucumber bacterial wilt',
    'cucumber powdery mildew', 'eggplant cercospora leaf spot',
    'eggplant phomopsis fruit rot', 'eggplant phytophthora blight',
    'garlic leaf blight', 'garlic rust', 'ginger leaf spot', 'ginger sheath blight',
    'grape black rot', 'grape downy mildew', 'grape leaf spot',
    'grapevine leafroll disease', 'lettuce downy mildew', 'lettuce mosaic virus',
    'maple tar spot', 'peach anthracnose', 'peach brown rot', 'peach leaf curl',
    'peach rust', 'peach scab', 'plum bacterial spot', 'plum brown rot',
    'plum pocket disease', 'plum pox virus', 'plum rust', 'potato early blight',
    'potato late blight', 'raspberry fire blight', 'raspberry gray mold',
    'raspberry leaf spot', 'raspberry yellow rust', 'rice blast', 'rice sheath blight',
    'soybean bacterial blight', 'soybean brown spot', 'soybean downy mildew',
    'soybean frog eye leaf spot', 'soybean mosaic', 'soybean rust',
    'squash powdery mildew', 'strawberry anthracnose', 'strawberry leaf scorch',
    'tobacco blue mold', 'tobacco brown spot', 'tobacco frogeye leaf spot',
    'tobacco mosaic virus', 'tomato bacterial leaf spot', 'tomato early blight',
    'tomato late blight', 'tomato leaf mold', 'tomato mosaic virus',
    'tomato septoria leaf spot', 'tomato yellow leaf curl virus',
    'wheat bacterial leaf streak (black chaff)', 'wheat head scab', 'wheat leaf rust',
    'wheat loose smut', 'wheat powdery mildew', 'wheat septoria blotch',
    'wheat stem rust', 'wheat stripe rust', 'zucchini bacterial wilt',
    'zucchini downy mildew', 'zucchini powdery mildew', 'zucchini yellow mosaic virus',
)
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255
assert NUM_CLASSES == 116
CLASS_HASH = hashlib.sha256(json.dumps(CLASS_NAMES).encode()).hexdigest()


def palette() -> np.ndarray:
    """Unique deterministic display palette, not the label-ID encoding."""
    colors = np.zeros((256, 3), dtype=np.uint8)
    for j in range(256):
        lab = j
        for k in range(8):
            for ch in range(3):
                colors[j, ch] |= ((lab >> ch) & 1) << (7 - k)
            lab >>= 3
    colors[IGNORE_INDEX] = (180, 180, 180)
    return colors
