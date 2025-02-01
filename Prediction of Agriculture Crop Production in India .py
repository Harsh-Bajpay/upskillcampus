import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="India Agriculture Crop Production Predictor",
    page_icon="🌾",
    layout="wide"
)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    # Load data
    df = pd.read_csv('India Agriculture Crop Production.csv')
    
    # Preprocessing steps
    df['Year'] = df['Year'].str[:4].astype(int)
    df = df.dropna()
    df = df.drop('Area Units', axis=1)
    
    # Convert production units
    df.loc[df['Crop'] == 'Cotton(lint)','Production'] = (df.loc[df['Crop'] == 'Cotton(lint)','Production']*170)/1000
    df.loc[df['Crop'] == 'Jute','Production'] = (df.loc[df['Crop'] == 'Jute','Production']*180)/1000
    df.loc[df['Crop'] == 'Mesta','Production'] = (df.loc[df['Crop'] == 'Mesta','Production']*180)/1000
    df.loc[df['Crop'] == 'Coconut','Production'] = (df.loc[df['Crop'] == 'Coconut','Production']*1.5)/1000
    
    df = df.drop(['Production Units', 'Yield'], axis=1)
    df['Yield'] = df['Production'] / df['Area']
    
    return df

@st.cache_resource
def train_model(df):
    """Train the LightGBM model"""
    # Feature preprocessing
    df['Area'] = np.log1p(df['Area'])
    df['Production'] = np.log1p(df['Production'])
    
    # Handle rare categories
    columns = ['District', 'Crop']
    labels = {'District': 'diğer1', 'Crop': 'diğer2'}
    for i in columns:
        alt_frekans_degeri = df[i].value_counts().quantile(0.05)
        frekanslar = df[i].value_counts()
        nadir_kategorikler = frekanslar[frekanslar < alt_frekans_degeri].index
        df[i] = df[i].replace(nadir_kategorikler, labels[i])
    
    # Remove outliers and prepare features
    df = df.drop(['State', 'Year', 'Yield'], axis=1)
    df = pd.get_dummies(df, columns=['District', 'Season', 'Crop'], drop_first=True)
    
    # Split features and target
    X = df.drop('Production', axis=1)
    y = df['Production']
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=17)
    
    # Train model
    model = lgb.LGBMRegressor(
        boosting_type='gbdt',
        num_leaves=40,
        learning_rate=0.17,
        n_estimators=30,
        max_depth=30,
        random_state=33
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l2')
    
    return model

def predict_production(model, district, season, area, crop=None):
    """Make prediction for single crop or all crops"""
    area = np.log1p(area)
    
    if crop:
        # Single crop prediction
        input_data = pd.DataFrame({
            'District': [district],
            'Season': [season],
            'Area': [area]
        })
        
        input_data = pd.get_dummies(input_data, columns=['District', 'Season'], drop_first=True)
        model_features = model.booster_.feature_name()
        
        missing_cols = list(set(model_features) - set(input_data.columns))
        missing_data = pd.DataFrame(0, index=input_data.index, columns=missing_cols)
        input_data = pd.concat([input_data, missing_data], axis=1)
        
        input_data = input_data[model_features]
        
        crops = [col for col in model_features if 'Crop_' in col]
        for col in crops:
            input_data[col] = 1 if col == f'Crop_{crop}' else 0
            
        prediction = model.predict(input_data)[0]
        return np.exp(prediction) - 1
    else:
        # All crops prediction
        input_data = pd.DataFrame({
            'District': [district],
            'Season': [season],
            'Area': [area]
        })
        
        input_data = pd.get_dummies(input_data, columns=['District', 'Season'], drop_first=True)
        model_features = model.booster_.feature_name()
        
        missing_cols = list(set(model_features) - set(input_data.columns))
        missing_data = pd.DataFrame(0, index=input_data.index, columns=missing_cols)
        input_data = pd.concat([input_data, missing_data], axis=1)
        
        input_data = input_data[model_features]
        
        crops = [col for col in model_features if 'Crop_' in col]
        results = {}
        
        for crop in crops:
            input_data_crop = input_data.copy()
            for col in crops:
                input_data_crop[col] = 1 if col == crop else 0
            prediction = model.predict(input_data_crop)
            results[crop.replace('Crop_', '')] = np.exp(prediction[0]) - 1
            
        return results

def main():
    st.title("🌾 India Agriculture Crop Production Predictor")
    
    # Load data and train model
    with st.spinner("Loading data and training model..."):
        df = load_and_preprocess_data()
        model = train_model(df)
    
    # Sidebar
    st.sidebar.header("Input Parameters")
    
    # Get unique values
    districts = sorted(df['District'].unique())
    seasons = sorted(df['Season'].unique())
    crops = sorted(df['Crop'].unique())
    
    # Input fields
    district = st.sidebar.selectbox("Select District", districts)
    season = st.sidebar.selectbox("Select Season", seasons)
    area = st.sidebar.number_input("Enter Area (hectares)", min_value=1, value=100)
    
    # Prediction type
    pred_type = st.sidebar.radio("Prediction Type", ["Single Crop", "All Crops"])
    
    if pred_type == "Single Crop":
        crop = st.sidebar.selectbox("Select Crop", crops)
        if st.sidebar.button("Predict Production"):
            prediction = predict_production(model, district, season, area, crop)
            
            st.success(f"Predicted production for {crop}: {prediction:.2f} tonnes")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(['Predicted Production'], [prediction])
            ax.set_ylabel('Production (tonnes)')
            ax.set_title(f'Predicted Production for {crop}')
            st.pyplot(fig)
            
    else:
        if st.sidebar.button("Predict Production for All Crops"):
            predictions = predict_production(model, district, season, area)
            
            # Display results
            st.success("Production Predictions for All Crops")
            
            # Convert predictions to DataFrame for better display
            pred_df = pd.DataFrame(predictions.items(), columns=['Crop', 'Predicted Production'])
            pred_df = pred_df.sort_values('Predicted Production', ascending=False)
            
            # Display top 10 crops
            st.subheader("Top 10 Crops by Predicted Production")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=pred_df.head(10), x='Predicted Production', y='Crop')
            plt.title('Top 10 Crops by Predicted Production')
            st.pyplot(fig)
            
            # Display full results in a table
            st.subheader("All Predictions")
            st.dataframe(pred_df)
    
    # Model performance metrics
    st.sidebar.markdown("---")
    if st.sidebar.checkbox("Show Model Performance"):
        st.subheader("Model Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test Score", f"{model.best_score_['valid_0']['l2']:.4f}")
        with col2:
            st.metric("Number of Features", len(model.feature_name()))

if __name__ == "__main__":
    main()
