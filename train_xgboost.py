import os
import pickle
import numpy as np
from xgboost import XGBClassifier


def train_xgboost_binary(X_train, y_train):
    """
    Train Binary IDS using XGBoost (modern boosting model)
    """

    os.makedirs("models", exist_ok=True)

    # Handle class imbalance (VERY IMPORTANT for IDS)
    pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Save XGBoost model
    with open("models/xgboost_binary_ids.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ XGBoost Binary IDS trained and saved as models/xgboost_binary_ids.pkl")

    return model
