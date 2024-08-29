import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Load the pre-trained model and scaler
scaler = StandardScaler()
iso_forest = IsolationForest(contamination=0.02, random_state=42)

# Assume you have fit the scaler and model on the training data

def main():
    st.title("Credit Card Fraud Detection")

    uploaded_file = st.file_uploader("Choose a file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Preprocess
        df = preprocess_data(df)
        df[['Amount', 'Time']] = scaler.transform(df[['Amount', 'Time']])
        features = df[[f'V{i}' for i in range(1, 29)]]
        df['anomaly'] = iso_forest.predict(features)
        df['anomaly'] = df['anomaly'].apply(lambda x: 'Fraudulent' if x == -1 else 'Non-Fraudulent')

        # Display results
        st.write(df)
        st.write(f"Number of Fraudulent Transactions: {df['anomaly'].value_counts().get('Fraudulent', 0)}")

if __name__ == "__main__":
    main()
