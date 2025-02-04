# Analyzing-House-Sale-Prices-in-King-County-USA

This project analyzes house sale prices in King County, USA (which includes Seattle) and builds predictive models to estimate the price of a house based on its features. The dataset covers homes sold between May 2014 and May 2015. In this project you will:

- **Import and clean data:** Remove unwanted columns, convert date strings to datetime objects, and handle missing values.
- **Perform exploratory data analysis (EDA):** Generate summary statistics, create visualizations (boxplots and regression plots) to explore relationships between features and the sale price.
- **Develop models:** Build linear regression models using single and multiple features, and implement a pipeline that scales, transforms, and models the data.
- **Evaluate and refine models:** Split the data into training and testing sets, then apply Ridge regression (with and without a polynomial transformation) to assess the model performance.

## Project Structure

- **main.py:** Contains the full Python code for data preparation, exploratory analysis, model development, and evaluation. The code is commented to explain each step.
- **README.md:** Provides an overview of the project, instructions, and a description of the contents.

## Requirements 

Make sure you have the following Python libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- requests

## Running the Project
Clone this repository.
Make sure the required packages are installed.
Run the main.py file:
bash
Copy
Edit
python main.py
The script will:

Download the dataset (if not already present)
Display data types and summary statistics
Save visualizations (e.g., boxplot_waterfront_price.png and regplot_sqft_above_price.png) in the project folder
Print model performance (R² scores) for various regression models to the console

