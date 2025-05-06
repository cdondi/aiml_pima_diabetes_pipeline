# Project: Data Pipeline with DVC and MLflow for Machine Learning

This project demonstrates how to build an end-to-end machine learning pipeline using **DVC (Data Version Control)** for data and model versioning, and **MLflow** for experiment tracking. The pipeline focuses on training a **Random Forest Classifier** on the **Pima Indians Diabetes Dataset**, with clear stages for data preprocessing, model training, and evaluation.

## Key Features

### Data Version Control (DVC)
- Tracks and versions datasets, models, and pipeline stages for reproducibility across environments.
- Pipeline is structured into modular stages (preprocessing, training, evaluation) that automatically re-execute when dependencies change.
- Supports remote storage options (e.g., DagsHub, S3) for large datasets and models.

### Experiment Tracking with MLflow
- Tracks experiment parameters, metrics, and artifacts.
- Logs model hyperparameters (e.g., `n_estimators`, `max_depth`) and evaluation metrics such as accuracy.
- Supports comparison of different model runs to guide optimization.

## Pipeline Stages

### Preprocessing
- `preprocess.py` reads the raw dataset (`data/raw/data.csv`), applies basic transformations (e.g., renaming columns), and writes processed data to `data/processed/data.csv`.
- Ensures consistent data preparation across runs.

### Training
- `train.py` trains a Random Forest Classifier on the processed dataset.
- Model is saved to `models/random_forest.pkl`.
- Hyperparameters and the trained model are logged to MLflow.

### Evaluation
- `evaluate.py` loads the trained model and evaluates accuracy on the dataset.
- Evaluation results are logged to MLflow.

## Goals
- **Reproducibility**: DVC ensures that data, code, and parameters reproduce consistent results.
- **Experimentation**: MLflow supports easy tracking and comparison of experiments.
- **Collaboration**: DVC and MLflow enable organized team workflows with clear versioning.
- **Feature Engineering**: We are not doing any feature engineering for this project. The goal is to build a reproducible pipeline.

## Use Cases
- **Data Science Teams**: Maintain structured, reproducible pipelines and model histories.
- **Machine Learning Research**: Rapidly iterate, manage data versions, and track results over time.

## Technology Stack
- **Python**: Core language for scripts and model code.
- **DVC**: Manages data, models, and pipeline stage versions.
- **MLflow**: Tracks metrics, parameters, and model artifacts.
- **Scikit-learn**: Used for training the Random Forest Classifier.

This project showcases how to manage the entire lifecycle of a machine learning project in a reproducible, trackable, and collaborative way.

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize DVC and pull data
dvc init
dvc pull

# Run the full pipeline
dvc repro
```

## DVC Stage Commands

```bash
# Preprocessing stage
dvc stage add -n preprocess \
    -p preprocess.input,preprocess.output \
    -d src/preprocess.py -d data/raw/data.csv \
    -o data/processed/data.csv \
    python src/preprocess.py

# Training stage
dvc stage add -n train \
    -p train.data,train.model,train.random_state,train.n_estimators,train.max_depth \
    -d src/train.py -d data/raw/data.csv \
    -o models/model.pkl \
    python src/train.py

# Evaluation stage
dvc stage add -n evaluate \
    -d src/evaluate.py -d models/model.pkl -d data/raw/data.csv \
    python src/evaluate.py
```