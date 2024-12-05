# Pixel-Level Interpretability in Medical Imaging: A Fusion of Machine Learning and Fuzzy Logic

This repository contains the code and resources for reproducing the experiments described in the manuscript. The project focuses on pixel-level interpretability (PLI) powered by fuzzy logic to enhance diagnostic transparency in medical imaging.

---

## 1. Overview
This repository supports:
- **Preprocessing**: Scripts to handle real-world medical imaging datasets.
- **Training**: Implementation of the PLI model using both simulated and real-world data.
- **Heatmap Generation**: Scripts to produce interpretability maps.
- **Reproducibility**: Steps to validate metrics and replicate results from the manuscript.

---

## 2. Requirements
To run the scripts, you need the following dependencies:
- **Python >= 3.8**
- **TensorFlow >= 2.x**
- **NumPy**
- **OpenCV**
- **Matplotlib**
- **Pandas**
- **Scikit-learn**
## 3. Dataset Information
The study utilizes publicly available chest X-ray datasets. Download the datasets from:

- **NIH Chest X-rays Dataset**
- **COVID-19 Radiography Database**
- **CheXpert Dataset**
- **Preprocessing**

## Preprocessing steps include:
- **Resizing images to 224x224 pixels.**
- **Normalizing pixel values.**
- **Reducing noise using Gaussian filtering.**
- **Applying data augmentation techniques such as rotation and flipping.**
## To preprocess the data, run:
Install dependencies using:
```bash
pip install -r requirements.txt
python preprocessing/preprocess_real_data.py --input_path data/raw --output_path data/processed
python train_real_data.py --data_path data/processed --batch_size 64 --epochs 30 --learning_rate 0.0001
python train_simulation.py --data_path data/simulation --batch_size 32 --epochs 20
python generate_pli_heatmaps.py --model_path models/pli_model.h5 --image_path data/processed/sample_image.jpg --output_path results/
python evaluate_metrics.py --data_path data/processed --model_path models/pli_model.h5
python generate_calibration_curve.py --data_path data/processed --model_path models/pli_model.h5
Adjust parameters such as learning rates, epochs, and fuzzification thresholds in configs/parameters.json. Example parameters include:
{
  "learning_rate": 0.0001,
  "batch_size": 64,
  "epochs": 30,
  "fuzzification_threshold": 0.5
}
python validate_model.py --data_path data/test --model_path models/pli_model.h5
python generate_logs.py --output_path logs/
```
## Troubleshooting
- **Dataset Issues: Ensure datasets are downloaded and paths are specified correctly in the scripts.**
- **Memory Errors: Reduce batch size or use a GPU for training to handle large datasets.**
- **Script Errors: Check if required dependencies are installed and compatible with your system.**
