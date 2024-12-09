import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def build_readmission_model(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Separate features (X) and target variable (y)
    X = df.iloc[:, :-1]  # All columns except the last one
    y = df.iloc[:, -1]   # Last column (readmission status)
    print(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the logistic regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': abs(model.coef_[0])
    }).sort_values('Importance', ascending=False)
    
    # Generate evaluation metrics
    print("\nModel Performance:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 10 Most Important Features')
    plt.show()
    
    return model, scaler, feature_importance

def predict_readmission(model, scaler, new_data):
    """
    Make predictions for new patients using the trained model
    
    Parameters:
    model: Trained logistic regression model
    scaler: Fitted StandardScaler
    new_data: DataFrame with the same features as training data (before dummies)
    
    Returns:
    Predictions and probabilities
    """
    # Prepare the new data same way as training data
    new_data_processed = pd.get_dummies(new_data)
    
    # Ensure new data has same columns as training data
    missing_cols = set(model.feature_names_in_) - set(new_data_processed.columns)
    for col in missing_cols:
        new_data_processed[col] = 0
        
    # Reorder columns to match training data
    new_data_processed = new_data_processed[model.feature_names_in_]
    
    # Scale the data
    new_data_scaled = scaler.transform(new_data_processed)
    
    # Make predictions
    predictions = model.predict(new_data_scaled)
    probabilities = model.predict_proba(new_data_scaled)
    
    return predictions, probabilities

build_readmission_model('Dataset/diabetic_data_training.csv')