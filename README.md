# Faster R-CNN + SVM Hybrid for Moving Object Recognition

This repo contains a reference implementation of the paper:

> **Semantic Analysis System to Recognize Moving Objects by Using a Deep Learning Model**  
> (IEEE Access, 2024)

It combines **Faster R-CNN** for region proposals + feature extraction, and a **Linear SVM** for refined classification of detected objects.  
Optional semantic priors (co-occurrence analysis) are included for improved recognition.

---

## 🔧 Requirements

Tested with:
- Python 3.10+
- PyTorch 2.2+, TorchVision 0.17+
- scikit-learn
- OpenCV
- pycocotools
- matplotlib
- tqdm

Install dependencies:

```bash
pip install torch torchvision scikit-learn opencv-python pycocotools tqdm numpy matplotlib joblib
