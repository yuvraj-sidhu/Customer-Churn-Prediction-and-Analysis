import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    data = pd.read_csv(path)
    return data

def preprocess_data(data):
    # Drop irrelevant columns, handle missing values, etc.
    data.dropna(inplace=True)
    # Dummy encoding for categorical features
    data = pd.get_dummies(data, drop_first=True)
    return data

if __name__ == "__main__":
    data = load_data("data/churn_data.csv")
    processed_data = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(
        processed_data.drop("Churn", axis=1), 
        processed_data["Churn"], 
        test_size=0.2, 
        random_state=42
    )
