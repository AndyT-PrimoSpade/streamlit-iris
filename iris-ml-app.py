import os
import pickle
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.base import ClassifierMixin
from typing import Tuple, Dict, Any

# App Constants
SEPAL_LENGTH_RANGE: Tuple[float, float] = (4.0, 8.5)
SEPAL_WIDTH_RANGE: Tuple[float, float] = (1.5, 5.0)
PETAL_LENGTH_RANGE: Tuple[float, float] = (0.5, 7.5)
PETAL_WIDTH_RANGE: Tuple[float, float] = (0.1, 3.0)

st.set_page_config(page_title="Iris Flower Prediction App")

# Title and description
st.title("Iris Flower Prediction App")
st.markdown("This app predicts the **Iris flower** type based on your input parameters!")

# Sidebar for user input parameters
st.sidebar.header("User Input Parameters")

def user_input_features() -> pd.DataFrame:
    sepal_length: float = st.sidebar.slider('Sepal length', *SEPAL_LENGTH_RANGE, 5.4, help="Select the sepal length in cm")
    sepal_width: float = st.sidebar.slider('Sepal width', *SEPAL_WIDTH_RANGE, 3.4, help="Select the sepal width in cm")
    petal_length: float = st.sidebar.slider('Petal length', *PETAL_LENGTH_RANGE, 1.3, help="Select the petal length in cm")
    petal_width: float = st.sidebar.slider('Petal width', *PETAL_WIDTH_RANGE, 0.2, help="Select the petal width in cm")
    
    return pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    })

@st.cache_resource
def load_models() -> Tuple[Dict[str, ClassifierMixin], Any]:
    """Loads the Iris dataset and models, training and saving them if necessary."""
    iris = load_iris()
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist

    model_filenames = {
        'Random Forest': os.path.join(model_dir, 'random_forest.pkl'),
        'K-Nearest Neighbors': os.path.join(model_dir, 'knn.pkl'),
        'Logistic Regression': os.path.join(model_dir, 'logistic_regression.pkl'),
        'Support Vector Machine': os.path.join(model_dir, 'svm.pkl')
    }
    
    models: Dict[str, ClassifierMixin] = {}
    
    for name, filename in model_filenames.items():
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                models[name] = pickle.load(file)
        else:
            if name == 'Random Forest':
                model = RandomForestClassifier()
            elif name == 'K-Nearest Neighbors':
                model = KNeighborsClassifier()
            elif name == 'Logistic Regression':
                model = LogisticRegression(max_iter=200)
            elif name == 'Support Vector Machine':
                model = svm.SVC(probability=True)
            
            model.fit(iris.data, iris.target)
            models[name] = model
            
            with open(filename, 'wb') as file:
                pickle.dump(model, file)
    
    return models, iris

# Load models and Iris data
models, iris_data = load_models()

# Get user input and display it
user_features: pd.DataFrame = user_input_features()
st.subheader("User Input Parameters")
st.write(user_features)

# Display predictions and probabilities with extra spacing between columns
st.subheader("Model Predictions")

# Define columns with spacing
for model_name, model in models.items():
    col1, col_space, col2 = st.columns([3, 1, 3])  # Adjust column ratios for spacing

    # Display each model's prediction and probability
    with col1:
        st.markdown(f"#### {model_name}")
        prediction = model.predict(user_features)
        predicted_class: str = iris_data.target_names[prediction][0]
        st.write(f"**Predicted Iris Class:** {predicted_class}")

    with col2:
        prediction_proba = model.predict_proba(user_features)
        prediction_proba_df: pd.DataFrame = pd.DataFrame(prediction_proba, columns=iris_data.target_names)
        st.write("**Prediction Probability:**")
        st.write(prediction_proba_df)