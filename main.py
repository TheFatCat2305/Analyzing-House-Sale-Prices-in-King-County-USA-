
---

### main.py

```python
#!/usr/bin/env python3
"""
House Sales in King County, USA
--------------------------------
This script loads and cleans the King County house sales dataset,
performs exploratory data analysis (EDA), develops predictive models,
and evaluates them. It includes:
  - Data import and wrangling
  - Visualization (boxplots and regression plots)
  - Model development using Linear Regression and Ridge Regression
  - A Pipeline with scaling, polynomial feature expansion, and regression
  - Evaluation using train/test splits
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def download_dataset(url: str, filename: str) -> None:
    """
    Download the dataset from a URL if it is not already present.

    Parameters:
        url (str): URL to download the file from.
        filename (str): Local filename to save the downloaded file.
    """
    if not os.path.exists(filename):
        print(f"Downloading dataset from {url} ...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            print("Download completed.")
        else:
            raise Exception("Failed to download dataset.")
    else:
        print(f"Dataset '{filename}' already exists. Skipping download.")


def main():
    # ============================
    # Module 1: Importing Data
    # ============================
    
    # URL of the dataset and local filename
    dataset_url = ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/'
                   'IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/'
                   'data/kc_house_data_NaN.csv')
    filename = "housing.csv"
    
    # Download the dataset if necessary
    download_dataset(dataset_url, filename)
    
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)
    
    # Convert the 'date' column to datetime format (errors are coerced to NaT)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Display the data types of each column
    print("Data Types of the DataFrame:")
    print(df.dtypes)
    print("\nStatistical Summary of the DataFrame:")
    print(df.describe())
    
    
    # ============================
    # Module 2: Data Wrangling
    # ============================
    
    # Drop the 'Unnamed: 0' and 'id' columns since they are not needed
    df.drop(["Unnamed: 0", "id"], axis=1, inplace=True)
    print("\nDataFrame after dropping columns 'Unnamed: 0' and 'id':")
    print(df.describe())
    
    # Check for missing values in 'bedrooms' and 'bathrooms'
    print("\nMissing Values Before Replacement:")
    print("Bedrooms:", df['bedrooms'].isnull().sum())
    print("Bathrooms:", df['bathrooms'].isnull().sum())
    
    # Replace missing values in 'bedrooms' with the mean of the column
    bedrooms_mean = df['bedrooms'].mean()
    df['bedrooms'].replace(np.nan, bedrooms_mean, inplace=True)
    
    # Replace missing values in 'bathrooms' with the mean of the column
    bathrooms_mean = df['bathrooms'].mean()
    df['bathrooms'].replace(np.nan, bathrooms_mean, inplace=True)
    
    # Verify that there are no more missing values
    print("\nMissing Values After Replacement:")
    print("Bedrooms:", df['bedrooms'].isnull().sum())
    print("Bathrooms:", df['bathrooms'].isnull().sum())
    
    
    # ============================
    # Module 3: Exploratory Data Analysis
    # ============================
    
    # Count the number of houses for each unique 'floors' value
    print("\nCount of Houses by Number of Floors:")
    print(df['floors'].value_counts().to_frame())
    
    # Create a boxplot to compare house prices with and without a waterfront view
    plt.figure()
    sns.boxplot(x='waterfront', y='price', data=df)
    plt.title("House Price Distribution by Waterfront Status")
    plt.xlabel("Waterfront (0 = No, 1 = Yes)")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig("boxplot_waterfront_price.png")  # Save the plot as an image
    plt.close()
    
    # Create a regression plot to analyze the relationship between sqft_above and price
    plt.figure()
    sns.regplot(x='sqft_above', y='price', data=df, line_kws={"color": "red"})
    plt.title("Regression Plot: Price vs. Sqft Above")
    plt.xlabel("Square Footage Above Ground")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig("regplot_sqft_above_price.png")  # Save the plot as an image
    plt.close()
    
    # Show correlation of all features with price
    print("\nCorrelation of Features with Price:")
    print(df.corr()['price'].sort_values())
    
    
    # ============================
    # Module 4: Model Development
    # ============================
    
    # --- Linear Regression using 'long' as the feature ---
    X_long = df[['long']]
    Y = df['price']
    lm = LinearRegression()
    lm.fit(X_long, Y)
    print("\nR^2 for Linear Regression using 'long':", lm.score(X_long, Y))
    
    # --- Question 6: Linear Regression using 'sqft_living' as the feature ---
    lm.fit(df[['sqft_living']], Y)
    r2_sqft_living = lm.score(df[['sqft_living']], Y)
    print("R^2 for Linear Regression using 'sqft_living':", r2_sqft_living)
    
    # --- Question 7: Linear Regression using multiple features ---
    features = df[["floors", "waterfront", "lat", "bedrooms", "sqft_basement",
                   "view", "bathrooms", "sqft_living15", "sqft_above", "grade",
                   "sqft_living"]]
    lm.fit(features, Y)
    r2_multi = lm.score(features, Y)
    print("R^2 for Linear Regression with multiple features:", r2_multi)
    
    # --- Question 8: Pipeline with scaling, polynomial transformation, and linear regression ---
    pipeline_steps = [
        ('scale', StandardScaler()),
        ('polynomial', PolynomialFeatures(include_bias=False)),
        ('model', LinearRegression())
    ]
    pipe = Pipeline(pipeline_steps)
    pipe.fit(features, Y)
    r2_pipeline = pipe.score(features, Y)
    print("R^2 for the Pipeline model:", r2_pipeline)
    
    
    # ============================
    # Module 5: Model Evaluation and Refinement
    # ============================
    
    # Define feature names and separate the predictors and target variable
    feature_names = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement",
                     "view", "bathrooms", "sqft_living15", "sqft_above", "grade",
                     "sqft_living"]
    X = df[feature_names]
    Y = df['price']
    
    # Split the data into training and testing sets (15% test size)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)
    print("\nNumber of Test Samples:", x_test.shape[0])
    print("Number of Training Samples:", x_train.shape[0])
    
    # --- Question 9: Ridge Regression (alpha=0.1) ---
    ridge_model = Ridge(alpha=0.1)
    ridge_model.fit(x_train, y_train)
    yhat_ridge = ridge_model.predict(x_test)
    r2_ridge = ridge_model.score(x_test, y_test)
    print("R^2 for Ridge Regression (alpha=0.1):", r2_ridge)
    
    # --- Question 10: Ridge Regression with a Second-Order Polynomial Transformation ---
    poly = PolynomialFeatures(degree=2)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.fit_transform(x_test)
    ridge_model.fit(x_train_poly, y_train)
    yhat_poly = ridge_model.predict(x_test_poly)
    r2_ridge_poly = ridge_model.score(x_test_poly, y_test)
    print("R^2 for Ridge Regression with Polynomial Features (degree=2):", r2_ridge_poly)


