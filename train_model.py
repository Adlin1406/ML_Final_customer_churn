import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features

def train():

    df = load_and_clean_data("data/customer_churn_data.csv")
    df = create_features(df)

    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = RandomForestClassifier(class_weight="balanced")
    model.fit(X_train, y_train)

    # Save model
    pickle.dump(model, open("models/model.pkl", "wb"))
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))

    print("Model trained and saved!")

if __name__ == "__main__":
    train()