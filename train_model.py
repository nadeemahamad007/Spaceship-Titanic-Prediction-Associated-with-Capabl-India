from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train.csv"
TEST_PATH = BASE_DIR / "test.csv"
SUBMISSION_PATH = BASE_DIR / "submission.csv"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    passenger_split = data["PassengerId"].str.split("_", expand=True)
    data["GroupId"] = passenger_split[0]
    data["PassengerNumber"] = pd.to_numeric(passenger_split[1], errors="coerce")

    cabin_split = data["Cabin"].fillna("Unknown/0/U").str.split("/", expand=True)
    data["Deck"] = cabin_split[0]
    data["CabinNumber"] = pd.to_numeric(cabin_split[1], errors="coerce")
    data["Side"] = cabin_split[2]

    data["LastName"] = data["Name"].fillna("Unknown Unknown").str.split().str[-1]

    spend_columns = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for column in spend_columns:
        data[column] = data[column].fillna(0)

    data["TotalSpend"] = data[spend_columns].sum(axis=1)
    data["HasSpending"] = (data["TotalSpend"] > 0).astype(int)
    data["IsChild"] = data["Age"].fillna(data["Age"].median()) < 18

    data["CryoSleep"] = data["CryoSleep"].fillna(False).astype(str)
    data["VIP"] = data["VIP"].fillna(False).astype(str)
    data["IsChild"] = data["IsChild"].astype(str)

    return data.drop(columns=["PassengerId", "Cabin", "Name"], errors="ignore")


def build_pipeline() -> Pipeline:
    numeric_features = [
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "PassengerNumber",
        "CabinNumber",
        "TotalSpend",
    ]
    categorical_features = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "Deck",
        "Side",
        "GroupId",
        "LastName",
        "HasSpending",
        "IsChild",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_and_evaluate() -> Pipeline:
    train_df = pd.read_csv(TRAIN_PATH)
    engineered = add_features(train_df)

    X = engineered.drop(columns=["Transported"])
    y = engineered["Transported"].astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_valid)
    accuracy = accuracy_score(y_valid, predictions)

    print(f"Validation accuracy: {accuracy:.4f}")
    print()
    print("Classification report:")
    print(classification_report(y_valid, predictions, digits=4))

    return pipeline


def create_submission(pipeline: Pipeline) -> None:
    test_df = pd.read_csv(TEST_PATH)
    passenger_ids = test_df["PassengerId"].copy()

    test_features = add_features(test_df)
    predictions = pipeline.predict(test_features).astype(bool)

    submission = pd.DataFrame(
        {
            "PassengerId": passenger_ids,
            "Transported": predictions,
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    fitted_pipeline = train_and_evaluate()
    create_submission(fitted_pipeline)
