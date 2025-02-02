# Breast Density Classification Using Deep Learning

This repository contains my bachelor's thesis project for the Computer Science program at the Federal Institute of Brasília (IFB). The project focuses on developing a deep learning model to classify breast density in full-field digital mammograms (FFDM) according to the BI-RADS® standard. The goal is to assist radiologists in diagnosing breast cancer by automating density classification, a critical factor in early detection.

## 📖 Summary

Breast density classification is a key component in mammographic analysis, as dense breast tissue can obscure tumors and increase cancer risk. This project explores state-of-the-art convolutional neural networks (CNNs) to classify breast density into four BI-RADS® categories (A, B, C, D). The work includes dataset integration, preprocessing, model training, and performance evaluation, with a focus on improving generalization and reproducibility.

Key contributions:
- Integration of multiple public mammography datasets.
- Novel multi-class training approach using binary classifiers.
- Extensive hyperparameter tuning and preprocessing pipelines.
- Open-source implementation for reproducibility.

## 📊 Results 

TODO: Add results and analysis.

## 🚀 Running the project

### Installation

1. Clone the repository:

```bash
git clone https://github.com/loioladev/breast-density-classification.git  
cd breast-density-classification  
```

2. Install the required dependencies using `uv sync` or install each dependency found in the [pyproject file](./pyproject.toml) manually using `pip`.

3. Download and prepare datasets to use in the project (see [Datasets Preparation](./src/datasets/README.md)).

### How to run

After downloading the datasets, you can process them using the following command:

```bash
uv run main.py convert <dataset> <path_to_dir>
```

Where `<dataset>` is the dataset name (e.g., `inbreast`, `cbis-ddsm`, `miniddsm`) and `<path_to_dir>` is the path to the directory containing the dataset files. More information use the flag `--help`.

To train the model, first set the configurations in the [config file](./src/configs/config.yaml) and run the following command:

```bash
uv run main.py train -f <config_file>
```

Where `<config_file>` is the path to the configuration file. More information use the flag `--help`.

---

For questions and collaborations, please contact me at matheusloiolapinto@gmail.com.