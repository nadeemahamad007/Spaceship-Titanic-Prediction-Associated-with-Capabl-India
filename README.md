# Spaceship Titanic Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-f7931e?style=for-the-badge&logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-GitHub%20Ready-success?style=for-the-badge)

An end-to-end machine learning project for the **Spaceship Titanic** classification problem. This repository includes exploratory analysis, feature engineering, a reproducible local training pipeline, and submission file generation for the test dataset.

## Overview

The goal of this project is to predict whether a passenger was transported to another dimension during the Spaceship Titanic incident. The original work started as a notebook-based solution and was later improved into a cleaner project structure that is easier to understand, run, and showcase on GitHub.

## Problem Statement

Given passenger details such as home planet, cabin, age, VIP status, and onboard spending, the task is to build a classification model that predicts the boolean target `Transported`.

## Key Improvements

- Converted the Colab-style workflow into a local Python training script.
- Added structured feature engineering from `PassengerId`, `Cabin`, `Name`, and spending columns.
- Built a reusable preprocessing and modeling pipeline with `scikit-learn`.
- Added project documentation, dependency tracking, and GitHub-ready organization.

## Workflow

1. Load the training and test datasets.
2. Clean missing values and transform mixed data types.
3. Engineer additional predictive features.
4. Train a `RandomForestClassifier` using a preprocessing pipeline.
5. Evaluate model quality on a validation split.
6. Generate `submission.csv` for the test dataset.

## Feature Engineering

The improved pipeline creates additional signals from the raw data:

- `GroupId` and `PassengerNumber` from `PassengerId`
- `Deck`, `CabinNumber`, and `Side` from `Cabin`
- `LastName` from `Name`
- `TotalSpend` from service-related spending columns
- `HasSpending` to identify passengers with any spending activity
- `IsChild` derived from age

## Model And Result

- Model: `RandomForestClassifier`
- Validation accuracy: `0.7349`
- Evaluation includes a classification report printed by `train_model.py`

## Repository Contents

- `train.csv` - training dataset
- `test.csv` - test dataset
- `sample_submission.csv` - sample submission format
- `Spaceship_Titanic.ipynb` - exploratory notebook with improved markdown structure
- `train_model.py` - reproducible local training and submission pipeline
- `requirements.txt` - Python dependencies
- `LINKEDIN_POST.md` - ready-to-use LinkedIn summary

## Tech Stack

- Python
- pandas
- scikit-learn
- Jupyter Notebook

## How To Run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the training script:

```bash
python train_model.py
```

The script will train the model, print evaluation metrics, and create `submission.csv`.

## Project Structure

```text
Project Spaceship Problem/
|-- README.md
|-- LINKEDIN_POST.md
|-- Spaceship_Titanic.ipynb
|-- train_model.py
|-- requirements.txt
|-- train.csv
|-- test.csv
|-- sample_submission.csv
`-- Spaceship problem.docx
```

## Future Improvements

- Compare additional models such as XGBoost, CatBoost, or LightGBM
- Add cross-validation and hyperparameter tuning
- Export the trained model for later reuse
- Include richer visualizations and model interpretation

## Author
Nadeem Ahamad
Prepared as a Data Science Internship Project and Refined for GitHub Portfolio Presentation.
