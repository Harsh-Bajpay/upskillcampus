import kagglehub
import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class CropProductionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = ['State_Name', 'District_Name', 'Crop_Year', 'Season', 
                        'Crop', 'Area', 'Production']
        
    def load_and_preprocess_data(self):
        """Load and preprocess the agricultural dataset from Kaggle"""
        try:
            # Download dataset from Kaggle
            dataset_path = kagglehub.dataset_download("pyatakov/india-agriculture-crop-production")
            
            # Load the CSV file
            df = pd.read_csv(f"{dataset_path}/india_crop_production.csv")
            
            # Basic preprocessing
            df = df.rename(columns={
                'State_Name': 'State',
                'District_Name': 'District',
                'Crop_Year': 'Year'
            })
            
            # Handle missing values
            df = df.dropna()
            
            # Feature engineering
            df['Yield'] = df['Production'] / df['Area']
            df['Year'] = df['Year'].astype(int)
            
            return df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def prepare_features(self, df):
        """Prepare features for model training"""
        # Create dummy variables for categorical columns
        categorical_cols = ['State', 'District', 'Season', 'Crop']
        df_encoded = pd.get_dummies(df, columns=categorical_cols)
        
        # Select features for model
        X = df_encoded.drop(['Production', 'Yield'], axis=1)
        y = df_encoded['Production']
        
        return X, y

    def train_model(self, X_train, y_train):
        """Train the crop production prediction model"""
        # Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        self.model = rf_model

class StreamlitApp:
    def __init__(self):
        self.predictor = CropProductionPredictor()
        
    def run(self):
        st.title("India Agriculture Crop Production Prediction System")
        
        try:
            # Load data
            data = self.predictor.load_and_preprocess_data()
            if data is None:
                st.error("Failed to load dataset. Please check your internet connection and try again.")
                return
            
            # Sidebar navigation
            st.sidebar.title("Navigation")
            page = st.sidebar.radio("Select Page", 
                ["Overview", "Data Analysis", "Prediction", "Model Performance"])
            
            if page == "Overview":
                self.show_overview(data)
            elif page == "Data Analysis":
                self.show_analysis(data)
            elif page == "Prediction":
                self.show_prediction(data)
            else:
                self.show_model_performance(data)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")

    def show_overview(self, data):
        st.header("Dataset Overview")
        
        # Display basic statistics
        st.subheader("Dataset Statistics")
        st.write(data.describe())
        
        # Display unique values
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Unique States")
            st.write(data['State'].nunique())
            st.subheader("Unique Crops")
            st.write(data['Crop'].nunique())
        with col2:
            st.subheader("Year Range")
            st.write(f"{data['Year'].min()} - {data['Year'].max()}")
            st.subheader("Total Districts")
            st.write(data['District'].nunique())

    def show_analysis(self, data):
        st.header("Data Analysis")
        
        # Top producing states
        st.subheader("Top 10 Producing States")
        top_states = data.groupby('State')['Production'].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        top_states.plot(kind='bar', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Crop-wise production
        st.subheader("Crop-wise Production")
        crop_prod = data.groupby('Crop')['Production'].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        crop_prod.plot(kind='bar', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Yearly trend
        st.subheader("Production Trend Over Years")
        yearly_prod = data.groupby('Year')['Production'].sum()
        fig, ax = plt.subplots(figsize=(10, 6))
        yearly_prod.plot(kind='line', ax=ax)
        st.pyplot(fig)

    def show_prediction(self, data):
        st.header("Crop Production Prediction")
        
        # Input form
        state = st.selectbox("Select State", sorted(data['State'].unique()))
        district = st.selectbox("Select District", 
                              sorted(data[data['State'] == state]['District'].unique()))
        crop = st.selectbox("Select Crop", sorted(data['Crop'].unique()))
        season = st.selectbox("Select Season", sorted(data['Season'].unique()))
        area = st.number_input("Area (in hectares)", min_value=0.0)
        year = st.number_input("Year", min_value=2000, max_value=2030)
        
        if st.button("Predict"):
            # Prepare input data
            input_data = pd.DataFrame({
                'State': [state],
                'District': [district],
                'Year': [year],
                'Season': [season],
                'Crop': [crop],
                'Area': [area]
            })
            
            # Make prediction
            X, y = self.predictor.prepare_features(data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            self.predictor.train_model(X_train, y_train)
            
            # Transform input data
            input_encoded = pd.get_dummies(input_data, columns=['State', 'District', 'Season', 'Crop'])
            # Align input columns with training data
            input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
            
            prediction = self.predictor.model.predict(input_encoded)
            st.success(f"Predicted Production: {prediction[0]:.2f} tonnes")

    def show_model_performance(self, data):
        st.header("Model Performance")
        
        # Prepare data
        X, y = self.predictor.prepare_features(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train model
        self.predictor.train_model(X_train, y_train)
        
        # Make predictions
        y_pred = self.predictor.model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{rmse:.2f}")
        with col2:
            st.metric("MAE", f"{mae:.2f}")
        with col3:
            st.metric("R² Score", f"{r2:.2f}")
        
        # Plot actual vs predicted
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("Actual Production")
        plt.ylabel("Predicted Production")
        plt.title("Actual vs Predicted Production")
        st.pyplot(fig)

def main():
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()
