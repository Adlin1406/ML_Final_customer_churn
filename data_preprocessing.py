import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.drop("customerID", axis=1, inplace=True)

    return df