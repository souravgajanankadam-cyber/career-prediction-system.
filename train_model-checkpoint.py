import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
import pickle
import os

# Define relative file paths for portability
DATA_PATH = "PS2_Dataset.csv"
MODEL_SAVE_PATH = "career_model_data.pkl"

def main():
    print("Loading dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please make sure it is in the same directory.")
        
    df = pd.read_csv(DATA_PATH)
    
    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    # We will copy the raw data for storage in the pickle file
    df_raw = df.copy()
    
    # Separate features and target
    X = df.drop(columns=['Suggested Job Role'])
    y = df['Suggested Job Role']
    
    # Define column types
    binary_cols = ['self-learning capability?', 'Extra-courses did', 'Taken inputs from seniors or elders', 'worked in teams ever?', 'Introvert']
    rating_cols = ['reading and writing skills', 'memory capability score']
    nominal_cols = [
        'certifications', 'workshops', 'Interested subjects', 'interested career area',
        'Type of company want to settle in?', 'Interested Type of Books',
        'Management or Technical', 'hard/smart worker'
    ]
    numeric_cols = ['Logical quotient rating', 'hackathons', 'coding skills rating', 'public speaking points']
    
    print("Preprocessing data...")
    # Clean binary columns
    for col in binary_cols:
        X[col] = X[col].str.lower().str.strip().map({'yes': 1, 'no': 0}).fillna(0).astype(int)
        
    # Clean rating columns
    for col in rating_cols:
        X[col] = X[col].str.lower().str.strip().map({'poor': 0, 'medium': 1, 'excellent': 2}).fillna(1).astype(int)
        
    # Ordinal Encoder for nominal columns
    # We preserve the category list for each column so the Streamlit app can show them in dropdowns
    nominal_categories = {}
    for col in nominal_cols:
        # Standardize strings
        X[col] = X[col].str.strip()
        # Find unique values sorted
        unique_vals = sorted(X[col].unique())
        nominal_categories[col] = unique_vals
        
    encoder = OrdinalEncoder(
        categories=[nominal_categories[col] for col in nominal_cols],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    
    # Fit encoder and transform
    X_encoded_nominal = encoder.fit_transform(X[nominal_cols])
    X_processed = X.copy()
    for i, col in enumerate(nominal_cols):
        X_processed[col] = X_encoded_nominal[:, i]
        
    # Ensure numeric columns are correct type
    for col in numeric_cols:
        X_processed[col] = pd.to_numeric(X_processed[col]).fillna(0).astype(float)
        
    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train Random Forest Classifier
    print("Training RandomForestClassifier...")
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42)
    rf_model.fit(X_processed, y_encoded)
    
    # Train KNN for Case-Based Match (using cosine distance on processed features)
    print("Training NearestNeighbors (CBR)...")
    knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn_model.fit(X_processed)
    
    # Create descriptive dictionary of categorical mappings for Streamlit UI dropdown selectboxes
    categorical_mappings = {
        'binary': ['No', 'Yes'],
        'ratings': ['Poor', 'Medium', 'Excellent'],
        'nominal': nominal_categories
    }
    
    # Save everything into a single pickle file
    model_data = {
        'rf_model': rf_model,
        'knn_model': knn_model,
        'encoder': encoder,
        'label_encoder': label_encoder,
        'categorical_mappings': categorical_mappings,
        'feature_names': list(X.columns),
        'binary_cols': binary_cols,
        'rating_cols': rating_cols,
        'nominal_cols': nominal_cols,
        'numeric_cols': numeric_cols,
        'X_processed': X_processed,
        'y_encoded': y_encoded,
        'df_raw': df_raw
    }
    
    print(f"Saving model and mapping data to {MODEL_SAVE_PATH}...")
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model_data, f)
        
    print("Model training pipeline completed successfully!")

if __name__ == "__main__":
    main()
