import pickle


def load_binary_ids():
    with open("models/intrusion_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    return model, scaler, encoder, feature_names
