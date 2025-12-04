# StairVision YOLO Training Repository
## CS663 Mobile Vision Project - Training Pipeline

This repository contains all training-related code, datasets, and model artifacts for the StairVision stair detection system.

---

## 📂 Repository Contents

### Notebooks
- **`StairVision_YOLO_Training.ipynb`**: Main training notebook (100 epochs)
- **`StairVision_Enhanced_300Epochs.ipynb`**: Extended training (300 epochs)

### Scripts
- **`convert_pascal_to_yolo.py`**: Converts Pascal VOC XML annotations to YOLO format

### Dataset
- **RGB-D Stair Dataset** (400+ images)
- Located externally: `/Users/yeshwanthnani/Downloads/RGB-D stair dataset/`
- Train/Val/Test split: 280/60/60

---

## 🚀 Training Pipeline

### 1. Dataset Preparation
```bash
# Convert Pascal VOC XML to YOLO format
python convert_pascal_to_yolo.py <xml_dir> <output_dir>

# Expected output structure:
# dataset/
# ├── images/
# │   ├── train/
# │   ├── val/
# │   └── test/
# └── labels/
#     ├── train/
#     ├── val/
#     └── test/
```

### 2. Training in Google Colab

**Open Notebook:**
```
StairVision_YOLO_Training.ipynb
```

**Training Configuration:**
```python
from ultralytics import YOLO

# Load pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU
    project='runs/detect',
    name='stair_detection'
)

# Export to TFLite
model.export(format='tflite', imgsz=640)
```

**GPU Requirements:**
- Google Colab with T4 GPU (free tier)
- Training time: ~2-3 hours for 100 epochs

---

## 📊 Training Results

### Final Metrics
| Metric | Value |
|--------|-------|
| mAP@50 | 97.2% |
| Precision | 96.8% |
| Recall | 93.4% |
| F1-Score | 95.1% |

### Model Artifacts
- **PyTorch Model**: `runs/detect/train/weights/best.pt` (12MB)
- **TFLite Model**: `stair_yolo_best_float32.tflite` (12MB)
- **Training Curves**: `results.png`
- **Confusion Matrix**: `confusion_matrix.png`

---

## 🔄 TFLite Conversion

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Export to TFLite (Float32)
model.export(format='tflite', imgsz=640, int8=False)

# Output: stair_yolo_best_float32.tflite
```

**Model Specifications:**
- Input: `[1, 640, 640, 3]` Float32
- Output: `[1, 5, 8400]` Float32
- Size: 12MB
- Format: Float32 (no quantization)

---

## 📁 Dataset Information

### Source
- RGB-D Staircase Dataset
- 400+ images with bounding box annotations
- Pascal VOC XML format (converted to YOLO)

### Class Distribution
| Type | Count | Percentage |
|------|-------|------------|
| Ascending | 150 | 37.5% |
| Descending | 180 | 45% |
| Side View | 70 | 17.5% |

### Data Augmentation
- Horizontal flip: 50%
- Rotation: ±10°
- Scale: 0.1-1.9x
- HSV shift
- Mosaic augmentation

---

## 🛠️ Setup Instructions

### Prerequisites
```bash
pip install ultralytics
pip install opencv-python
pip install pandas matplotlib
```

### Google Colab Setup
1. Upload notebook to Google Drive
2. Open in Colab
3. Enable GPU: Runtime → Change runtime type → T4 GPU
4. Mount Google Drive
5. Run all cells

---

## 📈 Training Commands

**Basic Training:**
```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
```

**Extended Training (300 epochs):**
```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=300 imgsz=640 patience=50
```

**Validation:**
```bash
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml
```

---

## 🔗 Related Repository

**Android App Repository:**  
https://github.com/Yesh492/cs663-stair-detection-android

---

## 📝 Citation

```bibtex
@software{stairvision2024,
  title={StairVision: Real-Time Stair Detection for Visual Accessibility},
  author={Yeshwanth Nani},
  year={2024},
  course={CS663 Mobile Vision},
  institution={University},
  model={YOLOv8n},
  accuracy={97.2\%}
}
```

---

## 📧 Contact

**Project**: StairVision CS663  
**Author**: Yeshwanth Nani  
**Repository**: https://github.com/Yesh492/stair-detection-training

---

**Note**: This repository contains ONLY training-related files. For the Android application, see the `cs663-stair-detection-android` repository.
