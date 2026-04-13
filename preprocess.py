import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_train_test(train_df, test_df):
    # Separate features and labels
    X_train = train_df.drop(['label'], axis=1)
    y_train = train_df['label']
    X_test = test_df.drop(['label'], axis=1)
    y_test = test_df['label']

    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    numeric_cols = X_train.select_dtypes(exclude=['object']).columns

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]))
    X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical_cols]))

    # Reset column names for encoded data
    X_train_encoded.columns = encoder.get_feature_names_out(categorical_cols)
    X_test_encoded.columns = encoder.get_feature_names_out(categorical_cols)

    # Combine encoded categorical and numerical columns
    X_train_final = pd.concat([X_train_encoded, X_train[numeric_cols].reset_index(drop=True)], axis=1)
    X_test_final = pd.concat([X_test_encoded, X_test[numeric_cols].reset_index(drop=True)], axis=1)

    # 🔧 Fix: Convert column names to string to avoid sklearn 1.5+ TypeError
    X_train_final.columns = X_train_final.columns.astype(str)
    X_test_final.columns = X_test_final.columns.astype(str)

    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)

    feature_names = X_train_final.columns

    return X_train_scaled, y_train, X_test_scaled, y_test, scaler, encoder, feature_names
