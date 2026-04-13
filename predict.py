import pandas as pd
import joblib
import os
from tabulate import tabulate
from termcolor import colored

# Load saved model, scaler, encoder, and feature names
model = joblib.load("models/intrusion_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")  # <-- NEW
feature_names = joblib.load("models/feature_names.pkl")

# Define categorical columns (must match training)
categorical_cols = ["protocol_type", "service", "flag"]

def predict_from_csv(input_csv):
    if not os.path.exists(input_csv):
        print(f"❌ CSV file '{input_csv}' not found!")
        return
    
    # Load the CSV
    df = pd.read_csv(input_csv)

    # Ensure required columns exist
    missing_cols = set(feature_names) - set(df.columns)
    if missing_cols:
        print(f"❌ Missing columns in CSV: {missing_cols}")
        return

    # Separate categorical & numeric columns
    df_categorical = df[categorical_cols]
    df_numeric = df.drop(columns=categorical_cols)

    # Encode categorical columns
    df_encoded = encoder.transform(df_categorical)
    df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out(categorical_cols))

    # Merge numeric + encoded
    df_final = pd.concat([df_numeric.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)

    # Ensure correct column order
    df_final = df_final[feature_names]

    # Scale
    df_scaled = scaler.transform(df_final)

    # Predict
    predictions = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)

    # Prepare display data
    table_data = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        label = "Attack" if pred == 1 else "Normal"
        confidence = round(max(prob) * 100, 2)
        
        if label == "Attack":
            colored_label = colored(label, "red", attrs=["bold"])
        else:
            colored_label = colored(label, "green", attrs=["bold"])
        
        table_data.append([i+1, colored_label, f"{confidence}%"])

    # Print results table
    print("\n🔍 Predictions:")
    print(tabulate(table_data, headers=["Sample #", "Prediction", "Confidence"], tablefmt="fancy_grid"))

    # Save results to CSV
    df_results = pd.DataFrame({
        "Prediction": ["Attack" if p == 1 else "Normal" for p in predictions],
        "Confidence (%)": [round(max(prob) * 100, 2) for prob in probabilities]
    })
    os.makedirs("predictions", exist_ok=True)
    output_path = "predictions/new_predictions.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to {output_path}")

if __name__ == "__main__":
    csv_path = "new_inputs_extended.csv"  # Change to your CSV file name
    predict_from_csv(csv_path)
