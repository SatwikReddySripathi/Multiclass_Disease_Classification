# Multiclass_Disease_Classification
This project is designed to develop, deploy, and maintain a machine learning model for multiclass disease classification of chest X-rays. It utilizes a Machine Learning Operations (MLOps) framework to ensure seamless development, deployment, and continuous monitoring of the model. The project follows best practices for reproducibility, modularity, and scalability.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Information](#dataset-information)
3. [Project Workflow](#project-workflow)
4. [Prerequisites](#prerequisites)
5. [Git Repo and Project Structure](#git-repo-and-project-structure)
6. [Data Storage and Model Registry](#data-storage-and-model-registry)
7. [Pipeline](#pipeline)
8. [Application Interface](#application-interface)
9. [Monitoring Dashboard](#monitoring-dashboard)
10. [Contributors](#contributors)
11. [Acknowledgments](#acknowledgments)

## Introduction
Thoracic diseases, such as pneumonia, emphysema, and cardiomegaly, are significant health concerns globally, affecting millions of people each year. In the United States alone, respiratory diseases contribute to a high percentage of hospitalizations and healthcare costs. Accurate diagnosis of these conditions is crucial for effective treatment, yet traditional diagnostic methods rely heavily on manual analysis of medical images like chest X-rays. This process can be time-consuming and subject to human error, especially in areas with limited access to trained radiologists.

Motivated by these challenges, we chose this topic for our MLOps project. By leveraging the principles of MLOps, we have developed an end-to-end machine learning pipeline for automated thoracic disease classification using chest X-rays. Our solution is built to enhance the accuracy and efficiency of non-invasive diagnostic methods, aiming to assist healthcare providers with reliable, timely, and scalable diagnostic support.

By automating the process of classifying multiple thoracic diseases from chest X-ray images, this approach has the potential to alleviate the burden on healthcare professionals, reduce diagnostic errors, and provide faster results, especially in underserved areas where radiologists may not always be available.

This project integrates best practices in MLOps to ensure that our model is not only accurate but also easy to maintain and deploy, making it suitable for real-world clinical use.

## Dataset Information
The project utilizes the NIH ChestX-ray14 dataset, one of the largest publicly available chest X-ray image datasets for research purposes. The dataset is provided by the National Institutes of Health (NIH) Clinical Center and consists of 112,120 frontal-view chest X-ray images from 30,805 unique patients. The dataset is annotated with labels for 14 different thoracic disease categories, including:
### DataCard
- Dataset Name: NIH ChestX-ray14
- Source: National Institutes of Health (NIH) Clinical Center
- Link to Dataset: [NIH Clinical Center ChestX-ray Dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)
- Domain: Medical Imaging (Radiology)
- Task: Multiclass disease classification from chest X-ray images

### Disease Categories - Labels
1. Atelectasis 
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia

## Project Workflow
![Basic Project Workflow](assets/HighLevel_arch.png)
This is the basic Project Flow and this will be updated after the final system architecture is designed

## Git Repo and Project Structure
**Project Structure**
```
├── .dvc
│   ├── config                # Configuration file for DVC (Data Version Control)
│   ├── .gitignore            # Specifies files/folders related to DVC to be ignored by Git
├── data
│   ├── raw_notebooks              # Directory containing preprocessed training data
│   │   ├── Pre-processing.ipynb   #A Visual example of how the pre-processing steps are executed for a sample data
├── assets
│   ├── images 
├── config
│   ├── _init_.py
├── Dags
│   ├── Src
│   │   ├── Data  # Kubernetes deployment configuration for the frontend
│   │   |    ├── sampled_data.dvc
│   │   |    ├── sampled_data_entry.csv.dvc
│   │   ├── get_data_from_gcp.py
│   │   ├── cloud_data_management.py
│   │   ├── schema_generation.py
│   │   ├── anomaly_detection.py
│   │   ├── preprocessing.py
├── logs/scheduler
│   ├── model.py              # Defines the model architecture for training
│   ├── train.py              # Script for training the machine learning model
│   ├── evaluate.py           # Script for evaluating the model's performance
│   ├── predict.py            # Script for making predictions using the trained model
│   ├── requirements.txt      # Python dependencies required for the backend
├── .dvcignore                 # Specifies files/folders that should be ignored by DVC
├── .gitignore                 # Specifies files/folders that should be ignored by Git
├── dockerfile                 # Dockerfile to build the Docker image for the main application
├── README.md                  # read me file
├── docker-compose.yaml        # Docker Compose file to set up and run multi-container Docker applications
├── requirements.txt           # Python dependencies required for running the project locally
├── sampled_data.dvc
├── sampled_data_entry.csv.dvc



```
## Prerequisites 
#### 1. Softwares to be installed
Before you start, please make sure that the following are already installed in your machine, if not please install them.
- Git
- Docker
- Airflow
- DVC
- Python 3
- Pip

#### 2. Dependencies installation
```
pip install -r requirements.txt
```
#### 3. Clone the repositry
```
git clone https://github.com/SatwikReddySripathi/Multiclass_Disease_Classification.git
```

## Data Storage and Model Registry
## Pipeline
## Application Interface
## Monitoring Dashboard
## Contributors
[![MLOPs contributors](https://contrib.rocks/image?repo=SatwikReddySripathi/Multiclass_Disease_Classification)](https://github.com/SatwikReddySripathi/Multiclass_Disease_Classification/graphs/contributors)

* [@SatwikReddySripathi](https://github.com/SatwikReddySripathi)
* [@DhanushAkula](https://github.com/DhanushAkula)
* [@DhananJayKumarMV](https://github.com/DhananJayKumarMV)
* [@SravyaKodati](https://github.com/SravyaKodati)
* [@vamsijilla](https://github.com/vamsijilla)
* [@malkarsaidheeraj](https://github.com/malkarsaidheeraj)

## Acknowledgments
