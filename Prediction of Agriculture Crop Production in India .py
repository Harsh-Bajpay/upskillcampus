import kagglehub
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class CropProductionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self):
        """Load and preprocess the agricultural dataset from Kaggle"""
        try:
            # Download dataset from Kaggle
            dataset_path = kagglehub.dataset_download("pyatakov/india-agriculture-crop-production")
            
            # Load the CSV files
            df = pd.read_csv(f"{dataset_path}/india_crop_production.csv")
            
            # Handle missing values
            df = df.fillna(df.mean())
            
            # Feature engineering
            df['Season'] = pd.to_datetime(df['Year']).dt.month.map(
                lambda x: 'Kharif' if 6 <= x <= 10 else 'Rabi' if x <= 3 or x >= 11 else 'Zaid'
            )
            
            # Encode categorical variables
            df = pd.get_dummies(df, columns=['Season', 'State', 'Crop'])
            
            return df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    # ... rest of the class implementation remains the same ...

class StreamlitApp:
    def __init__(self):
        self.predictor = CropProductionPredictor()
        
    def run(self):
        st.title("Agriculture Crop Production Prediction System")
        
        try:
            # Load data
            data = self.predictor.load_and_preprocess_data()
            if data is None:
                st.error("Failed to load dataset. Please check your internet connection and try again.")
                return
                
            # Rest of the app implementation
            # ... existing code ...
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

def main():
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()
