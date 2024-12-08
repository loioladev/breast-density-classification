# Datasets for the project

This folder contains the scripts to prepare the datasets used in the project. The datasets are not included in this repository due to their size. For each dataset, there is a script do prepare it using the command `convert`. The download of the dataset is explained in this document, and should be done before running the scripts.

## Datasets

The datasets used in this project are:

- [InBreast](#inbreast)
- [RSNA](#rsna)

For all datasets, the metadata is stored in a CSV file with the following columns:
- `filename`: the name of the image file, usually with patient ID and study date informations
- `laterality`: the laterality of the breast, either `L` or `R`
- `view`: the view of the breast, either `CC` or `MLO`
- `density`: the breast density, either \[A-Z\], \[0-3\] or \[1-4\].

## InBreast

The InBreast dataset is a public dataset of full-field digital mammograms (FFDM). The dataset is available at the [Kaggle](https://www.kaggle.com/datasets/martholi/inbreast) platform. The database [InBreast](https://paperswithcode.com/dataset/inbreast) has a total of 115 cases (410 images) from which 90 cases are from women with both breasts affected (four images per case) and 25 cases are from mastectomy patients (two images per case). Several types of lesions (masses, calcifications, asymmetries, and distortions) were included. Accurate contours made by specialists are also provided in XML format, but not used in this project.

### Download

To download it, you need to create an account on Kaggle and download the [dataset](https://www.kaggle.com/datasets/martholi/inbreast) via the website or API.

### Preprocessing

The dataset is in DICOM format, so it is necessary to convert it to PNG. The script `inbreast.py` does this conversion. It converts the DICOM to PNG using the `pydicom` library, and a function to recort the breast from the image is applied. The data also passes a CLAHE filter to improve the contrast, as seen in this [paper](https://www.sciencedirect.com/science/article/pii/S2352340920308222).

## RSNA 

The RSNA dataset is a public dataset of full-field digital mammograms (FFDM), created for a cancer prediction competition in [Kaggle](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview). The goal of the competition was to identify cases of breast cancer in mammograms from screening exams, and many images have annotations of the breast density.

### Download

To download it, you need to create an account on Kaggle and download the [dataset](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview) via the website or API.

### Preprocessing

## VinDr Mammo

The [VinDr-Mammo](https://www.physionet.org/content/vindr-mammo/1.0.0/) dataset is a large-scale full-field digital mammography dataset, which can be used for the purpose of developing and evaluating algorithms for providing cancer assessment and breast density following the Breast Imaging Report and Data System (BI-RADS). The dataset is split into 1,000 test exams and 4,000 training exams, with the frequencies of each BI-RADS category, density level, and abnormality category being preserved by applying an iterative stratification algorithm.

### Download

Access [Physionet website](https://www.physionet.org/content/vindr-mammo/1.0.0/) and create an account to download the dataset. Use the `wget` command to download the dataset via terminal.

## Preprocessing

## Mini-DDSM

### Download

To download it, you need to create an account on Kaggle and download the [dataset](https://www.kaggle.com/datasets/cheddad/miniddsm2/data) via the website or API.

### Preprocessing

## BMCD

### Download

To download it, go to the [dataset website](https://zenodo.org/records/5036062) and download the `Dataset.zip` file.

### Preprocessing
