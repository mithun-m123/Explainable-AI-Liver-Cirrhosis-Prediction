import pandas as pd
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import train_test_split

def train():
    # Load Kaggle Cirrhosis Dataset
    df = pd.read_csv('datasets/cirrhosis.csv')
    df = df.dropna(subset=['Stage'])
    df = df.drop(columns=['ID', 'Status'], errors='ignore')
    df = df.dropna()

    X = pd.get_dummies(df.drop(columns=['Stage']), drop_first=True)
    y = df['Stage'].astype(int) - 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(eval_metric='mlogloss')
    model.fit(X_train, y_train)
    joblib.dump(model, 'backend/models/symptom_model.pkl')
    print("Symptom Model Saved!")

if __name__ == "__main__":
    train()