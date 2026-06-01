# Spaceship Titanic Prediction

This project predicts whether a passenger was transported to another dimension in the **Spaceship Titanic** machine learning challenge. The repository includes the original dataset files, an exploratory notebook, and a cleaner training pipeline that can be run locally without Google Colab.

## Project Highlights

- Reworked the original Colab-style workflow into a reproducible local Python script.
- Added feature engineering from `PassengerId`, `Cabin`, and spending-related columns.
- Built a preprocessing and modeling pipeline using `scikit-learn`.
- Generates a ready-to-upload `submission.csv` file for the test dataset.

## Dataset Files

- `train.csv` - training dataset with the `Transported` target
- `test.csv` - test dataset used for prediction
- `sample_submission.csv` - sample output format
- `Spaceship_Titanic.ipynb` - original exploratory notebook
- `train_model.py` - improved training and submission pipeline

## Feature Engineering Used

The enhanced pipeline creates additional signals from the raw dataset:

- `GroupId` and `PassengerNumber` from `PassengerId`
- `Deck`, `CabinNumber`, and `Side` from `Cabin`
- `LastName` from `Name`
- `TotalSpend` from all onboard service columns
- `HasSpending` to capture whether a passenger spent anything
- `IsChild` derived from `Age`

## Tech Stack

- Python
- pandas
- scikit-learn

## How To Run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the training pipeline:

```bash
python train_model.py
```

The script will:

- train a model on the training set
- print validation accuracy and a classification report
- generate `submission.csv`

## Project Structure

```text
Project Spaceship Problem/
├── README.md
├── Spaceship_Titanic.ipynb
├── train_model.py
├── train.csv
├── test.csv
├── sample_submission.csv
└── Spaceship problem.docx
```

## Suggested GitHub Repository Name

`spaceship-titanic-prediction`

## Ideas For Future Improvement

- compare multiple models such as XGBoost, LightGBM, or CatBoost
- add cross-validation and hyperparameter tuning
- save the trained model artifact for reuse
- move notebook insights into visual reports or a presentation

## Author

Prepared for internship/project submission and GitHub portfolio publishing.
