import time
import numpy as np

from load_model import load_binary_ids
from load_data import load_data
from preprocess import preprocess_train_test
from alert import trigger_alert


def simulate_live_detection(delay=1):
    """
    Simulates live network traffic by predicting
    one connection at a time.
    """

    print("\n🚀 Starting LIVE Intrusion Detection Simulation...\n")

    # Load trained binary IDS
    model, scaler, encoder, feature_names = load_binary_ids()

    # Load raw dataset again (for simulation only)
    train_df, test_df = load_data("KDDTrain+.txt", "KDDTest+.txt")

    # Preprocess (same pipeline as training)
    _, _, X_test, y_test, _, _, _ = preprocess_train_test(train_df, test_df)
    total_attacks = 0

    # Simulate live stream (one row at a time)
    for i in range(20):   # simulate first 20 connections
        sample = X_test[i].reshape(1, -1)
        prediction = model.predict(sample)[0]

        print(f"📡 Incoming connection #{i+1}")

        if prediction == 1:
            total_attacks += 1
            print("⚠️  Prediction: ATTACK")
            print(f"🚨 Total attacks detected so far: {total_attacks}")

            trigger_alert(1)

        else:
            print("✅ Prediction: NORMAL")

        print("-" * 50)
        time.sleep(delay)

    print("\n🛑 Live simulation ended.")


if __name__ == "__main__":
    simulate_live_detection(delay=1)
