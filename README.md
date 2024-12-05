# Pixel-Level Interpretability in Medical Imaging: A Fusion of Machine Learning and Fuzzy Logic

This repository contains the code and resources for reproducing the experiments described in the manuscript. The project focuses on using pixel-level interpretability (PLI) powered by fuzzy logic to enhance diagnostic transparency in medical imaging.

---

## 1. Overview
This repository supports:
- **Preprocessing**: Scripts to handle real-world medical imaging datasets.
- **Training**: Implementation of the PLI model using both simulated and real-world data.
- **Heatmap Generation**: Scripts to produce interpretability maps.
- **Reproducibility**: Steps to validate metrics and replicate results from the manuscript.

---

## 2. Requirements
Install the following dependencies:
- Python >= 3.8
- TensorFlow >= 2.x
- NumPy
- OpenCV
- Matplotlib
- Pandas
- Scikit-learn

Install dependencies using:
```bash
pip install -r requirements.txt
python preprocessing/preprocess_real_data.py --input_path data/raw --output_path data/processed
python train_real_data.py --data_path data/processed --batch_size 64 --epochs 30 --learning_rate 0.0001
python generate_pli_heatmaps.py --model_path models/pli_model.h5 --image_path data/processed/sample_image.jpg --output_path results/
python evaluate_metrics.py --data_path data/processed --model_path models/pli_model.h5
python validate_model.py --data_path data/test --model_path models/pli_model.h5
