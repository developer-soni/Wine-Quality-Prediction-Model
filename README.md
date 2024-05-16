# Wine-Quality-Prediction-Model
Quality Prediction Model using Random Forest


# Overview
This project aims to predict the quality of wine based on various chemical properties using a Random Forest classifier. The dataset used for this project contains several chemical attributes of wine and their respective quality ratings.

<img src="/wqp-model-heatmap.png">

# Dataset
The dataset used in this project contains the following features:

* Fixed Acidity
* Volatile Acidity
* Citric Acid
* Residual Sugar
* Chlorides
* Free Sulfur Dioxide
* Total Sulfur Dioxide
* Density
* pH
* Sulphates
* Alcohol
* Quality (target variable)
  \
  \
The data is sourced from the UCI Machine Learning Repository.

# Project Structure
* Wine_Quality_Prediction.ipynb: Jupyter Notebook containing the complete code for data exploration, preprocessing, model training, evaluation, and predictions.
* README.md: Project documentation.
* requirements.txt: List of Python libraries required to run the project.

# Installation
To run this project, you need to have Python installed. You can install the required libraries using pip:

```pip install -r requirements.txt```

Also create requirements.txt and add these dependencies listed below:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```


# Steps
1. Data Collection and Preparation
* Loading Data: The dataset is loaded using Pandas.
* Exploration: Initial exploration to understand the dataset structure and check for missing values.
2. Data Analysis and Visualization
* Statistical Summary: Using describe() to get statistical measures.
* Visual Analysis: Creating visualizations (histograms, bar plots) to explore relationships between features and the target variable.
* Correlation Matrix: Generating a heatmap to visualize correlations between features and quality.
3. Data Preprocessing
* Label Binarization: Converting quality ratings into binary labels (0 for bad quality, 1 for good quality).
* Feature-Target Split: Separating features (X) and target variable (y).
* Train-Test Split: Splitting the data into training and test sets using train_test_split.
4. Model Building
* Random Forest Classifier: Training a Random Forest model using the training data.
* Hyperparameter Tuning: Optionally tuning model parameters to optimize performance.
5. Model Evaluation
* Predictions: Using the model to make predictions on the test set.
* Accuracy Score: Calculating the accuracy of the model using accuracy_score.
6. Building a Predictive System
* Input Data: Preparing new input data for prediction.
* Prediction: Using the trained model to predict the quality of new wine samples.
* Output: Displaying the predicted quality.

# Usage
To use the predictive system, you need to input the chemical properties of the wine you want to predict. The model will output whether the wine is of good quality or not.

```
import numpy as np
import pandas as pd

# Define the input data as a list
input_data = [7.8, 0.88, 0.0, 2.6, 0.098, 25.0, 67.0, 0.9968, 3.2, 0.68, 9.8]

# Define the feature names
feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

# Create a DataFrame for the input data
input_data_df = pd.DataFrame([input_data], columns=feature_names)

# Load the trained model (replace 'model' with your trained model variable)
# model = ... (Assuming the model has been trained and is ready to use)

# Predict the quality using the trained model
prediction = model.predict(input_data_df)
print('Good Quality' if prediction[0] == 1 else 'Bad Quality')
```

# Conclusion
This project demonstrates the complete workflow of a machine learning project, from data collection and exploration to model building and deployment. By using a Random Forest classifier, the model achieves high accuracy in predicting wine quality based on chemical properties, showcasing the power of ensemble learning techniques.

# Acknowledgements
The dataset used in this project is sourced from the UCI Machine Learning Repository.
Special thanks to the original dataset creators for providing the data.

