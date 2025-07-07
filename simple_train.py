import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_and_save_model():
    """Train XGBoost model and save for deployment"""
    
    print("🚀 Starting model training...")
    
    # Load data
    df = pd.read_csv(r"E:\Personal Projects\Unicourt\MLops\diabetes-prediction-mlops\data\dataset_37_diabetes.csv")
    
    print(f"📊 Data loaded: {df.shape}")
    print(f"Target distribution:\n{df['class'].value_counts()}")
    
    # Preprocess data
    X = df.drop('class', axis=1)
    y = df['class'].map({'tested_positive': 1, 'tested_negative': 0})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Data preprocessing completed")
    
    # Train XGBoost model with optimized parameters
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=1.0,
        reg_alpha=0,
        reg_lambda=2,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )
    
    print("🔧 Training model...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"📈 Model Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and scaler
    print("💾 Saving model files...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save feature names
    feature_names = X.columns.tolist()
    with open('feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    
    print("✅ Model saved successfully!")
    print("Files created:")
    print("  - model.pkl (XGBoost model)")
    print("  - scaler.pkl (StandardScaler)")
    print("  - feature_names.txt (Feature names)")
    
    return model, scaler, accuracy, roc_auc

if __name__ == "__main__":
    model, scaler, accuracy, roc_auc = train_and_save_model()
    print(f"\n🎉 Training completed! Final accuracy: {accuracy:.4f}")