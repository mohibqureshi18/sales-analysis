# Cafe Sales Analysis & Automated Cleaning Tool

An end-to-end Data Science project that transforms "dirty" retail data into actionable insights through automated cleaning, exploratory data analysis (EDA), and interactive visualization.

## Live Dashboard

## Project Overview
This project addresses common data quality issues found in retail datasets (missing values, inconsistent formatting, and manual entry errors). I built a robust pipeline that:
1.  **Cleans** raw CSV data automatically.
2.  **Analyzes** sales trends using statistical visualizations.
3.  **Predicts** revenue trends using Machine Learning (Random Forest).
4.  **Deploys** the solution as a user-friendly Streamlit web app.

## Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn
* **Deployment:** Streamlit
* **Version Control:** Git & GitHub

## Key Features
* **Dynamic Data Upload:** Users can upload any CSV following the cafe sales schema.
* **Automated Imputation:** Handles 'ERROR', 'UNKNOWN', and missing price/quantity values using median and mode strategies.
* **Interactive Visuals:** * Correlation Heatmaps to find relationships between variables.
    * Date filters to drill down into specific months or weekdays.
    * Categorical analysis of Items, Payment Methods, and Locations.
* **Machine Learning:** Compares Linear Regression and Random Forest models to understand price elasticity and sales drivers.

## Repository Structure
```text
├── app.py                     # Streamlit web application code
├── PythonProject.ipynb         # Jupyter Notebook with full analysis & ML
├── requirements.txt            # List of dependencies for deployment
├── dirty_cafe_sales.csv        # Raw dataset for testing
└── README.md                   # Project documentation
