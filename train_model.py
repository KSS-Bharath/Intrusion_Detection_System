import os
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# ------------------------------------------------------------------
# Create models directory immediately when file loads (safe method)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_model(
    X_train,
    y_train,
    scaler=None,
    encoder=None,
    feature_names=None,
    sample_frac=1.0,
    model_name="multiclass_ids.pkl"
):
    """
    Trains Random Forest model with hyperparameter tuning.
    Saves trained model bundle safely inside models folder.
    """

    # --------------------------------------------------------------
    # Optional sampling (to reduce training time)
    # --------------------------------------------------------------
    if sample_frac < 1.0:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(X_train), int(len(X_train) * sample_frac), replace=False)

        X_train = X_train[indices]
        if hasattr(y_train, "iloc"):
            y_train = y_train.iloc[indices]
        else:
            y_train = y_train[indices]

    # --------------------------------------------------------------
    # Define Random Forest model
    # --------------------------------------------------------------
    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 20],
        "min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    # --------------------------------------------------------------
    # Train model
    # --------------------------------------------------------------
    grid.fit(X_train, y_train)

    print("\n✅ Hyperparameter tuning completed")
    print("✅ Best Parameters Found:")
    print(grid.best_params_)

    best_model = grid.best_estimator_

    # --------------------------------------------------------------
    # Save model bundle safely
    # --------------------------------------------------------------
    model_bundle = {
        "model": best_model,
        "scaler": scaler,
        "encoder": encoder,
        "feature_names": feature_names
    }

    save_path = os.path.join(MODELS_DIR, model_name)

    with open(save_path, "wb") as f:
        pickle.dump(model_bundle, f)

    print(f"✅ Model saved successfully at {save_path}")

    return best_model
