from train_xgboost import train_xgboost_binary
from load_data import load_data
from preprocess import preprocess_train_test
from train_model import train_model
from evaluate import evaluate_model
from alert import trigger_alert


from sklearn.metrics import accuracy_score, classification_report


def main():
    # ==================================================
    # 1. Load raw train and test data
    # ==================================================
    train_path = "KDDTrain+.txt"
    test_path = "KDDTest+.txt"

    train_df, test_df = load_data(train_path, test_path)

    # ==================================================
    # 2. Preprocess data (TRAIN + TEST handled here)
    # ==================================================
    (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
        encoder,
        feature_names
    ) = preprocess_train_test(train_df, test_df)

    # ==================================================
    # 3. MULTICLASS IDS
    # ==================================================
    print("\n================ TRAINING MULTICLASS IDS ================\n")

    model_multiclass = train_model(
        X_train,
        y_train,
        scaler=scaler,
        encoder=encoder,
        feature_names=feature_names,
        sample_frac=0.3
    )

    print("\n================ EVALUATING MULTICLASS IDS ================\n")
    evaluate_model(model_multiclass, X_test, y_test)

    # ==================================================
    # 4. BINARY IDS (Normal vs Attack)
    # ==================================================
    print("\n================ TRAINING BINARY IDS =====================\n")

    y_train_binary = (y_train != "normal").astype(int)
    y_test_binary = (y_test != "normal").astype(int)

    model_binary = train_model(
        X_train,
        y_train_binary,
        scaler=scaler,
        encoder=encoder,
        feature_names=feature_names,
        sample_frac=0.3
    )

    # ==================================================
    # 5. Evaluate Binary IDS
    # ==================================================
    y_pred_binary = model_binary.predict(X_test)

    print("\n================ BINARY IDS EVALUATION ==================\n")
    print(f"Binary Accuracy: {accuracy_score(y_test_binary, y_pred_binary):.4f}\n")
    print("Binary Classification Report:")
    print(classification_report(y_test_binary, y_pred_binary))

    # ==================================================
    # XGBOOST BINARY IDS (MODERN MODEL)
    # ==================================================
    print("\n================ TRAINING XGBOOST BINARY IDS ==================\n")

    xgb_model = train_xgboost_binary(X_train, y_train_binary)

    y_pred_xgb = xgb_model.predict(X_test)

    print("\n================ XGBOOST BINARY IDS EVALUATION ==================\n")
    print(f"XGBoost Accuracy: {accuracy_score(y_test_binary, y_pred_xgb):.4f}\n")
    print("XGBoost Classification Report:")
    print(classification_report(y_test_binary, y_pred_xgb))


    print("\n================ PIPELINE COMPLETED =====================\n")
    # ==================================================
    #  ATTACK ALERT LOGIC
    # ==================================================
    #trigger_alert(y_pred_binary)



if __name__ == "__main__":
    main()
